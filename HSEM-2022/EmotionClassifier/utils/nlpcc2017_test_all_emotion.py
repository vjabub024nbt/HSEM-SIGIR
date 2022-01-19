import os
import json
import random
random.seed(12)
emotions = ['other', 'like', 'sadness', 'disgust', 'anger', 'happiness']



f = open(os.path.join('../../ECM/data/NLPCC2017_95W/test.txt'), 'r', encoding='utf8')
post_writer = open(os.path.join('../../ECM/data/NLPCC2017_95W/manual_test_post.txt'), 'w', encoding='utf8')
sample_writer = [open(os.path.join(f'../../ECM/data/NLPCC2017_95W/manual_test_{emo}.txt'), 'w', encoding='utf8') for emo
                 in emotions]
sample = []
count = 0
data=f.readlines()
random.shuffle(data)
for line in data:
    post, post_emo, res, res_emo = line.strip().split('\t')
    if len(post.split()) > 5 and len(post.split()) < 12:
        sample.append([post, post_emo, res, res_emo])
        count += 1
    if count == 100:
        break

for post, post_emo, res, res_emo in sample:
    post_writer.write(post + '\n')
    for i, emo in enumerate(emotions):
        sample_writer[i].write(post + '\t' + post_emo + '\t' + res + '\t' + str(i) + '\n')

js = [open(os.path.join(f'../../ECM/data/NLPCC2017_95W/manual_test_{emo}.json'), 'w', encoding='utf8') for emo in
      emotions]

for i, j in enumerate(js):
    temp = []
    for post, post_emo, res, res_emo in sample:
        temp.append([post, post_emo, res, str(i)])
    json.dump(temp, j, ensure_ascii=False)
