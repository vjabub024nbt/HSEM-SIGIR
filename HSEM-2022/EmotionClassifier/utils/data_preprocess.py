"""
raw_data.txt -> train/valid/text.txt

include:
    - remove some special tokens such as "（转）（图片）//XXX@: 回复@XXX:"
    - tokenize
    - keep the top K most frequent words
"""
# coding=utf-8
import random
from config import train_rate, valid_rate, test_rate, SEED
from pyhanlp import HanLP
import re
import os


def clean_text(text):
    text = text.strip()
    # remove some special tokens such as "（转）（图片）//@XXX: 回复@XXX:"
    text = re.sub(r"（转）", r"", text)
    text = re.sub(r"（图）", r"", text)
    text = re.sub(r"//@.*?(:|：| )", r"", text)
    text = re.sub(r"回复@.*?(:|：| )", r"", text)
    text = re.sub(r"@.*?(:|：| )", r"", text)

    # tokenize
    res = []
    for term in HanLP.segment(text):
        res.append(term.word)
    return res


def write_to_file(token_list, label, writer):
    for i in range(len(token_list)):
        writer.write(token_list[i])
        if i != len(token_list) - 1: writer.write(' ')
    str2int = {'other': 0,
               'like': 1,
               'sadness': 2,
               'disgust': 3,
               'anger': 4,
               'happiness': 5}
    writer.write('\t' + str(str2int[label]) + '\n')
    # writer.write('\t' + str(label) + '\n')


if __name__ == '__main__':
    delete_fear_and_surprise = True  # 先设为false找到filter_other_rate
    filter_other_rate = 0.5
    min_length = 2
    max_length = 25

    random.seed(SEED)
    label_text_dict = {'anger': [],
                       'disgust': [],
                       'fear': [],
                       'happiness': [],
                       'like': [],
                       'sadness': [],
                       'surprise': [],
                       'other': []}

    with open(os.path.join('..', 'data', 'raw_data.txt'), 'r') as input \
            , open(os.path.join('..', 'data', 'train_3.txt'), 'w') as train \
            , open(os.path.join('..', 'data', 'dev_3.txt'), 'w') as valid \
            , open(os.path.join('..', 'data', 'test_3.txt'), 'w') as test:
        data = input.readlines()

        for line in data:
            text = line.strip().split('\t')[0].strip()
            label = line.strip().split('\t')[1].strip()

            text = clean_text(text)

            # 过滤掉清洗之后长度小于 min length 的句子
            if len(text) <= min_length or len(text) > max_length:
                continue

            label_text_dict[label].append(text)

        train_set = []
        test_set = []
        valid_set = []
        sum = 0

        print(f'label distribution after filter text length < {min_length}')

        for label, texts in label_text_dict.items():
            sum += len(texts)

        print(f'total sentences: {sum}')

        for label, texts in label_text_dict.items():
            print(f'{label}: {len(texts)}, {len(texts) / sum * 100:.1f}%')

            # 删除fear,surprise两个标签，因为比例太少
            if delete_fear_and_surprise:
                if label in ['fear', 'surprise']:
                    continue

            # 过滤一些other标签的句子，因为比例太大
            if label in ['other']:
                to_remove = []
                for _text in texts:
                    if random.random() < filter_other_rate:
                        to_remove.append(_text)
                for target in to_remove:
                    texts.remove(target)

            valid_num = int(len(texts) * valid_rate)
            test_num = int(len(texts) * test_rate)
            i = 0
            random.shuffle(texts)
            for _text in texts:
                if i < valid_num:
                    valid_set.append([label, _text])
                elif i < valid_num + test_num:
                    test_set.append([label, _text])
                else:
                    train_set.append([label, _text])
                i += 1

        print('-----------------------------------')
        print(f'train size: {len(train_set)}')
        print(f'valid size: {len(valid_set)}')
        print(f'test size: {len(test_set)}')
        print(f'total size: {len(train_set) + len(valid_set) + len(test_set)}')

        random.shuffle(train_set)
        # random.shuffle(valid_set)
        # random.shuffle(test_set)

        statistic = {'anger': [],
                     'disgust': [],
                     'fear': [],
                     'happiness': [],
                     'like': [],
                     'sadness': [],
                     'surprise': [],
                     'other': []}

        for data in train_set:
            writer = train
            label = data[0]
            text = data[1]
            write_to_file(text, label, writer)
            statistic[label].append(text)

        for data in valid_set:
            writer = valid
            label = data[0]
            text = data[1]
            write_to_file(text, label, writer)
            statistic[label].append(text)

        for data in test_set:
            writer = test
            label = data[0]
            text = data[1]
            write_to_file(text, label, writer)
            statistic[label].append(text)

        print('saved successfully')

        print('-------------------------------')

        print('label distribution after filter fear surprise other label')

        sum = 0
        for label, texts in statistic.items():
            sum += len(texts)

        print(f'total sentences: {sum}')

        for label, texts in statistic.items():
            print(f'{label}: {len(texts)}, {len(texts) / sum * 100:.1f}%')
