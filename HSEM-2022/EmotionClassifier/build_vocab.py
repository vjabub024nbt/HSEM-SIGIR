"""
use pretrained word embedding and train data to train text field and
label field and saved
"""
import os
import torch
from torchtext import data
from config import *
from utils.load_custom_embeddings import load_custom_embeddings
from transformers import BertTokenizer

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if use_bert:
    tokenizer = BertTokenizer.from_pretrained(bert_pretrained_name)


def tokenize_and_cut(sentence):
    max_input_length = tokenizer.max_model_input_sizes[bert_pretrained_name]
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


def build_vocab(vocab_size,
                path='data',
                train='train.txt'):
    print('start building vocab')

    if not use_bert:
        TEXT = data.Field(include_lengths=True)
    else:
        print('use pretrained bert tokenizer')
        TEXT = data.Field(batch_first=True,
                          use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=tokenizer.convert_tokens_to_ids,
                          init_token=tokenizer.cls_token_id,
                          eos_token=tokenizer.sep_token_id,
                          pad_token=tokenizer.pad_token_id,
                          unk_token=tokenizer.unk_token_id,
                          include_lengths=True)
    LABEL = data.LabelField()
    """
    fields definition:
        - the first element of these inner tuples will become the batch object's attribute name
        - the second element is the Field name
    """
    fields = [('text', TEXT), ('label', LABEL)]

    print('start create training dataset...')
    train_data = data.TabularDataset.splits(
        path=path,
        train=train,
        format='tsv',
        fields=fields
    )

    train_data = train_data[0]

    if not use_bert:
        custom_embeddings = load_custom_embeddings()
        TEXT.build_vocab(train_data,
                         max_size=vocab_size,
                         vectors=custom_embeddings
                         )

    LABEL.build_vocab(train_data)

    print(LABEL.vocab.stoi)
    print(tokenizer.vocab)

    if not use_bert:
        print(f'vocab size: {len(TEXT.vocab)}')
    else:
        print(f'bert vocab size: {len(tokenizer.vocab)}')
    print(f'label size: {len(LABEL.vocab)}')

    print('saving....')
    torch.save(TEXT, os.path.join(data_path, text_field))
    torch.save(LABEL, os.path.join(data_path, label_field))
    print('success saved')


if __name__ == '__main__':
    build_vocab(vocab_size, data_path, train_file)
