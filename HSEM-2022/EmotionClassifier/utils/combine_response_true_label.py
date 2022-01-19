"""
response.txt + emotion label in test_uniq_src.txt -> response_with_true_label.txt
"""
import os
import torch
from config import data_path, label_field

def combine(response_path, test_uniq_src, save_path):
    res = open(response_path, 'r', encoding='utf8').readlines()
    src = open(test_uniq_src, 'r', encoding='utf8').readlines()
    saver = open(save_path, 'w', encoding='utf8')

    assert len(res) == len(src)

    # 先把数据集中的标签转换成字符串
    num2str = ['other', 'like', 'sadness', 'disgust', 'anger', 'happiness']

    # 把字符串转换成分类器中的数字词典表示
    LABEL = torch.load(os.path.join('..',data_path, label_field))
    print(LABEL.vocab.stoi)

    for i in range(len(res)):
        response = res[i].strip("\n").replace(' </s>', '')
        true_label = int(src[i].strip('\n').split('\t')[3])
        print(response + '\t' + num2str[true_label])
        saver.write(response + '\t' + num2str[true_label] + '\n')


if __name__ == '__main__':
    # 数据集
    """ESTC"""
    # test_uniq_src = '../../ECM/data/ESTC/test_uniq_src.txt'
    """NLPCC2017"""
    # test_uniq_src = '../../ECM/data/NLPCC2017/valid.txt'
    """NLPCC2017_95W"""
    test_uniq_src = '../../ECM/data/NLPCC2017_95W/test.txt'

    # 生成的回复
    """MemGM NLPCC2017"""
    # response_path = '../../MemoryAugDialog/result/NLPCC2017_best_no_emotion'
    # save_path = '../../MemoryAugDialog/result/NLPCC2017_best_with_true_emotion'
    """MemGM ESTC"""
    # response_path = '../../MemoryAugDialog/result/ESTC_best_no_emotion'
    # save_path = '../../MemoryAugDialog/result/ESTC_best_with_true_emotion'

    """ECM NLPCC2017"""
    # response_path = '../../ECM/result/NLPCC2017/130000_with_no_emo_ppl'
    # save_path = '../../ECM/result/NLPCC2017/130000_with_emo_ppl'

    """ECM ESTC"""
    # response_path = '../../ECM/result/ESTC/seq2seq_with_no_emo'
    # save_path = '../../ECM/result/ESTC/seq2seq_with_emo'

    """ECM NLPCC2017_95W"""
    for i in range(1,18):
        response_path = f'../../ECM/result/NLPCC2017_95W/ecm_no_pretrain_goon/epoch_{i}_with_no_emo'
        save_path = f'../../ECM/result/NLPCC2017_95W/ecm_no_pretrain_goon/epoch_{i}_with_emo'

        combine(response_path, test_uniq_src, save_path)
