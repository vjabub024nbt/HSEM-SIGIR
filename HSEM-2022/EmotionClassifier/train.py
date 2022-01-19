"""
build vocab and train the model
"""
import os

from config import *
from utils.evaluate import *
from model import RNN, BERTLSTMSentiment
from utils.dataloader import get_data_iterator
import torch.optim as optim
import torch
import torch.nn as nn
import time
from transformers import BertTokenizer, BertModel
from build_vocab import tokenize_and_cut

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # model.train() is used to put the model in "training mode", which turns on dropout and batch normalization

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """
        total secs -> n mints m secs
    :param start_time:
    :param end_time:
    :return:
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda', gpuid)

    # get data_iterator of train/valid/test
    print('start getting data_iterator of train/valid/test...')
    TEXT = torch.load(os.path.join(data_path, text_field))
    LABEL = torch.load(os.path.join(data_path, label_field))
    train_iterator, valid_iterator, test_iterator = get_data_iterator(device,
                                                                      TEXT,
                                                                      LABEL,
                                                                      batch_size,
                                                                      data_path,
                                                                      train_file,
                                                                      valid_file,
                                                                      test_file)
    print('=================== success ===================')

    # build model
    print('start building model...')
    if not use_bert:
        input_dim = len(TEXT.vocab)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    output_dim = len(LABEL.vocab)

    if not use_bert:
        model = RNN(input_dim,
                    embedding_dim,
                    hidden_dim,
                    output_dim,
                    n_layers,
                    bidirectional,
                    dropout,
                    PAD_IDX)
    else:
        bert = BertModel.from_pretrained(bert_pretrained_name)

        print('load bert model successfully.')

        model = BERTLSTMSentiment(bert,
                                  hidden_dim,
                                  output_dim,
                                  n_layers,
                                  bidirectional,
                                  dropout)
        # freeze bert model's parameters
        for name, param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False

    print(f'The model has {count_parameters(model):,} trainable parameters')

    if not use_bert:
        # load pre-trained embeddings
        embeddings = TEXT.vocab.vectors

        print(f'pretrained_embeddings shape: {embeddings.shape}')
        model.embedding.weight.data.copy_(embeddings)

        # zero the initial weights of the unknown and padding tokens
        model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)

    # set optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    print('=================== success ===================')

    # train model
    print('start training model...')

    train_recoder = open(os.path.join('log', f'train_{mark}.txt'), 'w')
    train_recoder.write('train_loss\ttrain_acc\tvalid_loss\tvalid_acc\n')

    best_valid_loss = float('inf')
    best_epoch = 0

    # freeze embeddings
    if not use_bert:
        model.embedding.weight.requires_grad = False

    for epoch in range(epochs):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        train_recoder.write(f'{train_loss:.3f}\t{train_acc * 100:.2f}\t{valid_loss:.3f}\t{valid_acc * 100:.2f}\n')

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(model, os.path.join('saved_models', f'model_best.pkl'))
        if epoch - best_epoch > 10:
            print('valid loss did not reduce for a long time')
            break
        # else:
        #     # unfreeze embeddings
        #     model.embedding.weight.requires_grad = True

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'Best epoch: {best_epoch + 1:02} |  Best Val. loss: {best_valid_loss :.3f}')
        print('------------------------------------------------')

    print('test result')
    model = torch.load(os.path.join('saved_models', f'model_best.pkl'), map_location=device)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print('test_loss, test_acc', test_loss, test_acc)
