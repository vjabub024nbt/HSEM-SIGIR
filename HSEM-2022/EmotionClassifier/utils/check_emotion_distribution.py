"""
check emotion distribution on test/valid/train dataset
if they are imbalance, redistribute them.
"""

import os


def print_f(name, emotion_count, total):
    print(f'{name} emotion distribution:')
    print(f'other: {emotion_count[0]}\t{emotion_count[0] / total * 100:.2f}%')
    print(f'like: {emotion_count[1]}\t{emotion_count[1] / total * 100:.2f}%')
    print(f'sadness: {emotion_count[2]}\t{emotion_count[2] / total * 100:.2f}%')
    print(f'disgust: {emotion_count[3]}\t{emotion_count[3] / total * 100:.2f}%')
    print(f'anger: {emotion_count[4]}\t{emotion_count[4] / total * 100:.2f}%')
    print(f'happiness: {emotion_count[5]}\t{emotion_count[5] / total * 100:.2f}%')


def count_emotion_ratio(path, dataset):
    print(f'starting count emotion ratio in {dataset} dataset...\n')
    f = open(os.path.join(path, dataset + '.txt'), 'r', encoding='utf8')
    lines = f.readlines()
    """
    0: Other 
    1: Like 
    2: Sadness 
    3: Disgust 
    4: Anger 
    5: Happiness
    """
    src_emo_count = [0] * 6
    tgt_emo_count = [0] * 6

    for line in lines:
        src, src_emo, tgt, tgt_emo = line.strip().split('\t')
        src_emo_count[int(src_emo)] += 1
        tgt_emo_count[int(tgt_emo)] += 1

    total = len(lines)
    print_f('src', src_emo_count, total)
    print('')
    print_f('tgt', tgt_emo_count, total)
    print('==============================================')


if __name__ == '__main__':
    count_emotion_ratio('../../ESTC', 'test')
    count_emotion_ratio('../../ESTC', 'valid')
    count_emotion_ratio('../../ESTC', 'train')
    count_emotion_ratio('../../NLPCC2017', 'train')
    count_emotion_ratio('../../NLPCC2017', 'valid')
