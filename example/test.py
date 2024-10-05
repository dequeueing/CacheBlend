import json

json_number = input('input json number: \n')

# for sample_idx in range(2,3):
#     # parse the input
#     f = open(f"../inputs/{sample_idx}.json")
#     ex = json.load(f)
#     chunk_num = ex['chunk_num']
    
#     # print(chunk_num)
    
#     doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
#     q_prompt = ex['query']
    
#     for i in range(chunk_num):
#         print(doc_prompts[i])
        
#     print(q_prompt)

f = open(f"../inputs/{json_number}.json")
ex = json.load(f)
chunk_num = ex['chunk_num']

# print(chunk_num)

doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
q_prompt = ex['query']

for i in range(chunk_num):
    print(doc_prompts[i])
    
print(q_prompt)