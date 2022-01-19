import json
import random

random.seed(8888)

f = json.load(open('../data/nlpcc2017/raw_data.json', 'r', encoding='utf8'))

data = []

emo_count=[0]*6
for pair in f:
    post = pair[0][0].strip()
    response = pair[1][0].strip()
    emo_count[pair[1][1]]+=1
    len_post = len(post.split())
    len_response = len(response.split())

    if len_post <= 3 or len_post > 25 or len_response <= 3 or len_response > 25:
        continue

    data.append([post, response])

print(emo_count)
random.shuffle(data)








