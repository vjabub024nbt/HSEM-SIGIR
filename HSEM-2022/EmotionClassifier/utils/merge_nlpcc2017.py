import json
import os
f1 = json.load(open('../data/nlpcc2017/anotate/train_data.json', 'r', encoding='utf8'))
f2 = json.load(open('../data/nlpcc2017/anotate/test_data.json', 'r', encoding='utf8'))
f3 = json.load(open('../data/nlpcc2017/anotate/valid_data.json', 'r', encoding='utf8'))
print(len(f1))
print(len(f2))
print(len(f3))
data = []
for i in [f1, f2, f3]:
    for j in i:
        data.append([j[0][0], j[0][1], j[1][0], j[1][1]])

print(len(data))
print(data[:5])
js = open(os.path.join('../data/nlpcc2017/anotate/total_data.json'), 'w', encoding='utf8')
json.dump(data, js, ensure_ascii=False)
