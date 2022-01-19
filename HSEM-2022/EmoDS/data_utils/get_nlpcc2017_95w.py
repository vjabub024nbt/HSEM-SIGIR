import os

word_count_dict=dict()
train_data=open('../../ECM/data/NLPCC2017_95W/train.txt','r',encoding='utf8')
test_data=open('../../ECM/data/NLPCC2017_95W/test.txt','r',encoding='utf8')
id2emotion={0: 'none', 1: 'like' ,2: 'sadness' ,3: 'disgust' ,4: 'anger' ,5: 'happiness'}

output_train_post=open('../data/NLPCC2017_95W/train.post.data.txt','w',encoding='utf8')
output_train_response=open('../data/NLPCC2017_95W/train.response.data.txt','w',encoding='utf8')
output_train_emotion=open('../data/NLPCC2017_95W/train.emotion.labels.txt','w',encoding='utf8')

output_test_post=open('../data/NLPCC2017_95W/test.post.data.txt','w',encoding='utf8')
output_test_response=open('../data/NLPCC2017_95W/test.response.data.txt','w',encoding='utf8')
output_test_emotion=open('../data/NLPCC2017_95W/test.emotion.labels.txt','w',encoding='utf8')

output_word_count=open('../data/NLPCC2017_95W/word.count.txt','w',encoding='utf8')

for line in train_data.readlines():
    post,post_emo,response,response_emo=line.strip().split('\t')
    output_train_post.write(post+'\n')
    output_train_response.write(response+'\n')
    output_train_emotion.write(id2emotion[int(response_emo)]+'\n')

    for word in post.strip().split():
        if word in word_count_dict.keys():
            word_count_dict[word]=word_count_dict[word]+1
        else:
            word_count_dict[word]=1
    for word in response.strip().split():
        if word in word_count_dict.keys():
            word_count_dict[word]=word_count_dict[word]+1
        else:
            word_count_dict[word]=1

for word,count in word_count_dict.items():
    output_word_count.write(word+'|||'+str(count)+'\n')

for line in test_data.readlines():
    post, post_emo, response, response_emo = line.strip().split('\t')
    output_test_post.write(post + '\n')
    output_test_response.write(response + '\n')
    output_test_emotion.write(id2emotion[int(response_emo)] + '\n')




