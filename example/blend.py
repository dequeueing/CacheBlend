# 导入必要的库
# AutoTokenizer是transformer而不是vllm的一部分，这一点自己要注意
from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer 

# 初始化LLM和tokenizer
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

#TODO (Jiayi): fix last len

# 处理输入：对于每一个样本，打开并解析对应的json文件，提取文档块和查询
for sample_idx in range(1,11):
    # 解析输入，得到chunk_num, doc_prompts（文本块）和 q_prompt(用户查询)
    f = open(f"inputs/{sample_idx}.json")
    ex = json.load(f)
    chunk_num = ex['chunk_num']
    doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
    q_prompt = ex['query']
    
    # 用tokenizer对文档块(doc_prompts)和查询(query)进行编码
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    # Create a sampling params object.
    # 设置文本生成的参数
    # temperature: 控制生成的多样性，值越大生成的文本越多样
    # max_tokens: 生成文本的最大长度
    # 这里的实例，temperature=0，max_tokens=1，表示生成的文本是确定的，且只有一个token
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    # cache_fuse_metadata是cacheblend在vllm的基础之上新的抽象，用于控制cache的行为？
    # 自己应该仔细看一下cache_fuse_metadata相关的源码
    # 在vscode中全局搜索cache_fuse_metadata即可找到相关的代码
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False

    # 问题：这里的s_start和s_end是什么？magic number？
    # gpt：s_start和s_end是用于控制文档块和查询的拼接的，这里的s_start和s_end是固定的，不会随着输入的变化而变化
    s_start_full = [733, 4138, 28793]
    s_start_len = len(s_start_full) + 1 # 4

    #s_start = [518, 25580, 29962]
    s_start = []
    s_start_1_len = len(s_start) + 1 # 1

    # 为什么写了两个s_end？
    s_end = [518, 29914, 25580, 29962]
    s_end = [733, 28748, 16289, 28793]
    s_end_len = len(s_end) # 4, 没有使用的变量
    
    # 原来的kv值
    old_kvs = []
    
    # s_start_len, s_start_1_len, last_len
    # s_start_full, s_start, s_end    

    # 初始化文档块id列表，并在每个块前后添加开始和结束标记
    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids] # 在每个块前添加start标记
    doc_chunk_ids = [s_start_full] + doc_chunk_ids                     # 新添加一个chunk作为开始
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]              # 末尾添加一个chunk，内容为query和end-token
    
    # 例子：
    # doc_chunk_ids = [[101, 102, 103], [201, 202, 203]] 
    # q_ids = [301, 302, 303]
    # 最终得到的：doc_chunk_ids = [[733, 4138, 28793], [101, 102, 103], [201, 202, 203], [301, 302, 303, 733, 28748, 16289, 28793]]
    # 含义：start_chunk, text_chunk, query_chunk + end_chunk

    last_len = len([q_ids+s_end])

    # 设置cache fuse参数
    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False

    # 设置参数
    num_layer = 32  # 是decoder的layer数量吗？
    chunk_past_key_values = []
    
    # 对于每一个chunk，生成本文并且从模型中提取KV值
    # Concatenate old KVs
    for i in range(len(doc_chunk_ids)):
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        llm.generate(prompts, sampling_params) # 先inference，从而在gpu中保留kvcache
        
        # 这里的layer是什么？decoder的layer吗？ 
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(num_layer):  #对于model的每一层
            past_key_values = llm_layers[j].self_attn.hack_kv #拿到之前的kv cache
            # 根据不同的chunk拿到它们对应的kv cache
            if i == 0: # 如果是第一个块（也就是start_chunk）
                temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

            # chunk_past_key_values是一个列表，每一个元素是一个列表，包含了key和value
            if i == 0:
                # i==0意味着是第一个块，直接添加到chunk_past_key_values中
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                # 否则，将key和value添加到对应layer的chunk_past_key_values中
                #pdb.set_trace()
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
        #print(temp_k.shape[0])
        # 注意：chunk_past_key_values包含了所有的chunk在所有layer的kv值
        # 具体地，chunk_past_key_values[layer_num][0(K)/1(V)][chunk_num] -> kv states of <chunk_num> at <layer_num>
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values # 保存到old_kvs
        
    input_ids = []

    # 将所有文档块的id链接起来，但是跳过每个text_chunk中的start部分
    # 目标：通过拼接所有文档块的 token ID 序列，构建一个完整的输入序列，准备传递给模型进行推理。
    for i in range(len(doc_chunk_ids)):
        if i == 0: #第一个文档块：完全保留，因为它需要包括起始标记。
            temp_ids = doc_chunk_ids[i]
        else:      # 后续文档块：从第二个文档块开始，跳过起始标记部分，直接取剩余的 token。
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    # 解码成完整的文本提示
    input_prompt = tokenizer.decode(input_ids)
 
    # 对比是否进行fusing所带来的性能提升
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['suffix_len'] = last_len
    
    output = llm.generate([input_prompt], sampling_params)
    print(f"Cached generation: {output[0].outputs[0].text}")
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")
