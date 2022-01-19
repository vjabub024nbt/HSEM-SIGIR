"""
annotate stc -> emotional stc
"""
import os
import torch
from config import *
import json
from transformers import BertTokenizer, BertModel
from build_vocab import tokenize_and_cut
import time

from train import epoch_time

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def predict_class(model, tokens, TEXT, device):
    indexed = [TEXT.vocab.stoi[t] for t in tokens]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    preds = model(tensor, length_tensor)
    max_preds = preds.argmax(dim=1)
    return max_preds.item()


def predict_class_bert(model, tokenizer, sentence, device):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [init] + tokenizer.convert_tokens_to_ids(tokens) + [eos]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    length_tensor = torch.LongTensor(length)
    preds = model(tensor, length_tensor)
    max_preds = preds.argmax(dim=1)
    res = max_preds.item()
    return res


if __name__ == '__main__':
    device = torch.device('cuda', gpuid)
    TEXT = torch.load(os.path.join(data_path, text_field))
    LABEL = torch.load(os.path.join(data_path, label_field))
    
    model = torch.load(os.path.join('saved_models', best_model), map_location=device)
    
    
    model.to(device)
    model.eval()
    if use_bert:
        tokenizer = BertTokenizer.from_pretrained(bert_pretrained_name)
        max_input_length = tokenizer.max_model_input_sizes[bert_pretrained_name]
        init = tokenizer.cls_token_id
        eos = tokenizer.sep_token_id

    emotion_map = {0: 0, 1: 1, 2: 3, 3: 5, 4: 2, 5: 4}
    emotion_count = [0] * 6
    data_path = './data/nlpcc2017'
    output_dir = './data/nlpcc2017/anotate'
    
    for data in ['train_data']:
        print(f'start predict {data} set...')
        dataset_list = []
        f = open(os.path.join(data_path, data + '.txt'), 'r', encoding='utf8')
        lines = f.read().split('\n')
        size = len(lines)
        print(f'totally {size} pairs of conversations.')
        count = 0
        start_time = time.time()
        for pair in lines:
            if len(pair.split('\t')) != 2:
                continue
            src, tgt = pair.split('\t')
            if not use_bert:
                src_emo = emotion_map[predict_class(model, src, TEXT, device)]
                tgt_emo = emotion_map[predict_class(model, tgt, TEXT, device)]
            else:
                src_emo = emotion_map[predict_class_bert(model, tokenizer, src, device)]
                tgt_emo = emotion_map[predict_class_bert(model, tokenizer, tgt, device)]
            emotion_count[tgt_emo] += 1
            dataset_list.append([[src, src_emo], [tgt, tgt_emo]])
            count += 1
            if count % 1000 == 0:
                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                print(f'\r finished {count / size * 100:.4f}%. Time: {epoch_mins} m {epoch_secs} s.', end='',
                      flush=True)

        print('start writing to file...')
        js = open(os.path.join(output_dir, data + '.json'), 'w', encoding='utf8')
        txt = open(os.path.join(output_dir, data + '.txt'), 'w', encoding='utf8')
        json.dump(dataset_list, js, ensure_ascii=False)
        for line in dataset_list:
            print(f'\r {str(line[0][0])}\t{str(line[0][1])}\t{str(line[1][0])}\t{str(line[1][1])}', end='', flush=True)
            txt.write(str(line[0][0]) + '\t' + str(line[0][1]) + '\t' + str(line[1][0]) + '\t' + str(line[1][1]) + '\n')
        print(f'\n{data} set preprocess done.')
        print(f'emotion count:')
        print(f'other:{emotion_count[0]}')
        print(f'like:{emotion_count[1]}')
        print(f'sad:{emotion_count[2]}')
        print(f'disgust:{emotion_count[3]}')
        print(f'anger:{emotion_count[4]}')
        print(f'happy:{emotion_count[5]}')
