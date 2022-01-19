import math

import numpy
import torch
import argparse
import kgdlg
import json
from torch import cuda
import progressbar
import kgdlg.utils.misc_utils as utils
import os
import bleu
import time


def indices_lookup(indices, fields):
    words = [fields['tgt'].vocab.itos[i] for i in indices]
    sent = ' '.join(words)
    return sent


def batch_indices_lookup(batch_indices, fields):
    batch_sents = []
    for sent_indices in batch_indices:
        sent = indices_lookup(sent_indices, fields)
        batch_sents.append(sent)
    return batch_sents


def inference_file(translator,
                   data_iter,
                   test_data_file,
                   test_out, fields):
    print('start decoding ...')
    emo_num2str = ['other', 'like', 'sadness', 'disgust', 'anger', 'happiness']

    print(test_out)

    count = 1

    tgt_total_loss, tgt_total_words = 0, 0
    pred_total_loss, pred_total_words = 0, 0
    emo_pred_correct = 0
    start_time = time.time()
    print(f'start time: {start_time}')
    with open(test_out, 'w', encoding='utf8') as tgt_file:
        bar = progressbar.ProgressBar()
        for batch in bar(data_iter):

            print(f'finished {count}/{len(data_iter)}', end='\r', flush=True)
            # print(f'finished {count}/{len(data_iter)}')
            count += 1
            # if count == 7:
            #     break
            ret, loss_data, n_words, pred_correct = translator.inference_batch(batch)
            emo_pred_correct += pred_correct
            tgt_total_loss += loss_data
            tgt_total_words += n_words

            pred_total_loss += ret['scores'][0][0].item()
            pred_total_words += len(ret['predictions'][0][0])
            # print(ret['scores'][0][0])
            # print(ret['predictions'])
            # print('words: ', n_words)
            # print('loss: ', loss_data)
            # print(ret['predictions'][0])
            batch_sents = batch_indices_lookup(ret['predictions'][0], fields)

            # print('ret["scores"]', ret['scores'][0])
            # print('len ret["scores"]', len(ret['scores'][0]))
            pred_emo = ret['pred_emo'].item()
            pred_emo = int(fields['tgt_emo'].vocab.itos[pred_emo])
            for sent in batch_sents:
                if count < 15:
                    print(sent + '\t' + emo_num2str[pred_emo])
                tgt_file.write(sent + '\t' + emo_num2str[pred_emo] + '\n')

        ppl = math.exp(tgt_total_loss / tgt_total_words)
        pred_ppl = math.exp(-pred_total_loss / pred_total_words)
        # print('tgt_total_loss: ', tgt_total_loss)
        # print('tgt_total_words: ', tgt_total_words)
        print('test tgt PPL: ', ppl)
        # print('pred_total_loss: ', pred_total_loss)
        # print('pred_total_words: ', pred_total_words)
        print('test pred PPL: ', pred_ppl)
        print('test emo pred acc: ', emo_pred_correct / len(data_iter))
        print(test_out)
        end_time = time.time()
        print(f'end time: {end_time}')
        print(f'total time: {end_time - start_time}, {(end_time - start_time) / len(data_iter)}/per sentence')


def make_test_data_iter(data_path, fields, device, opt):
    if opt.vae_type in [0, 1, 2, 3, 4, 5]:  # TODO it is better for 5 to use "tgt" from train dataset
        test_datasdet = kgdlg.IO.InferDataset(
            data_path=data_path,
            fields=[('src', fields["src"])])
    elif opt.vae_type in [6, 7, 8]:
        test_dataset = kgdlg.IO.TrainDataset(  # Train Dataset
            data_path=data_path,
            fields=[('src', fields["src"]),
                    ('src_emo', fields['src_emo']),
                    ('tgt', fields["tgt"]),
                    ('tgt_emo', fields['tgt_emo'])])

    test_data_iter = kgdlg.IO.OrderedIterator(
        dataset=test_dataset, device=device,
        batch_size=1, train=False, sort=False,
        sort_within_batch=True, shuffle=False)
    return test_data_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_data", type=str)
    parser.add_argument("-test_out", type=str)
    parser.add_argument("-config", type=str)
    parser.add_argument("-config_with_loaded_model", type=str)
    parser.add_argument("-config_from_local_or_loaded_model", type=int)
    parser.add_argument("-model", type=str)
    parser.add_argument("-vocab", type=str)
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)
    parser.add_argument("-beam_size", type=int)
    parser.add_argument("-decode_max_length", type=int)
    parser.add_argument("-use_mmi", type=int)

    args = parser.parse_args()
    if 1 == args.config_from_local_or_loaded_model:  # loaded model (from out_dir)
        opt = utils.load_hparams(args.config_with_loaded_model)
    elif 0 == args.config_from_local_or_loaded_model:  # local (./config.yml)
        opt = utils.load_hparams(args.config)
    opt.out_dir = os.path.dirname(args.model)

    if args.gpuid:
        if -1 == int(args.gpuid[0]):
            device = None
            opt.use_cuda = False
            opt.cluster_param_in_cuda = 0
        else:
            cuda.set_device(args.gpuid[0])
            device = torch.device('cuda', args.gpuid[0])
            opt.gpuid = int(args.gpuid[0])
            opt.use_cuda = True
    # print("use_cuda:", use_cuda, "device:", device)

    fields = kgdlg.IO.load_fields_from_vocab(
        torch.load(args.vocab))
    test_data_iter = make_test_data_iter(args.test_data, fields, device, opt)
    model = kgdlg.ModelConstructor.create_base_model(opt, fields)

    print('Loading parameters ...')
    # print('args.model:', args.model)

    if 0 == opt.load_model_mode_for_inference:
        model.load_checkpoint(args.model)
    elif 1 == opt.load_model_mode_for_inference:
        model.load_checkpoint_by_layers(args.model)
    if opt.use_cuda:
        model.set_paramater_to_cuda()
        # model = model.cuda()  # Main Set GPU

    translator = kgdlg.Inferer(model=model,
                               fields=fields,
                               beam_size=args.beam_size,
                               opt=opt,
                               n_best=1,
                               max_length=args.decode_max_length,
                               global_scorer=None,
                               use_mmi=args.use_mmi,
                               tgt_vocab=fields['tgt'].vocab)

    inference_file(translator, test_data_iter, args.test_data, args.test_out, fields)


if __name__ == '__main__':
    main()
