"""
test the model with lowest loss on valid
"""
from torch import nn
from torchtext import data
from config import gpuid, SEED, data_path, text_field, label_field
import torch
import os
from utils.evaluate import evaluate
from build_vocab import tokenize_and_cut

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# mark = 'estc_test_17'
# generate_mark = 'estc_test_17'

mark = 'test_18'
generate_mark = 'test_18'

if __name__ == '__main__':
    print(mark)
    device = torch.device('cuda', gpuid)

    # get data_iterator of train/valid/test
    TEXT = torch.load(os.path.join(data_path, text_field))
    LABEL = torch.load(os.path.join(data_path, label_field))

    fields = [('text', TEXT), ('label', LABEL)]

    model = torch.load(os.path.join('saved_models', f'bert_blstm_64.52.pkl'), map_location=device)
    criterion = nn.CrossEntropyLoss().to(device)

    for i in range(2, 50):
        modelID = str(i)
        """MemGM"""
        # path = os.path.join('..', 'MemoryAugDialog', 'result', f'res_gene_{mark}_tdv22_epoch{modelID}_{generate_mark}')
        path = os.path.join('..', 'HSEMEC-V4', 'result', f'res_gene_{mark}_tdv22_epoch{modelID}_{generate_mark}')
        # path = os.path.join('..', 'ECM', 'result', 'NLPCC2017_95W','ecm_no_pretrain_goon', f'epoch_{i}_with_emo')
        # path = os.path.join(f'../EmoDS/result5/result_epoch_{i}.txt')
        # print(path)

        # create dataset
        test_data = data.TabularDataset(
            path=path,
            format='tsv',
            fields=fields
        )

        # create iterator
        test_iterator = data.Iterator(
            dataset=test_data,
            sort_within_batch=True,
            sort_key=lambda x: len(x.text),
            batch_size=256,
            shuffle=False,
            device=device)

        test_loss, test_acc = evaluate(model, test_iterator, criterion)
        # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
        print(f'Emo-Acc')
        print(f'{test_acc * 100:.2f}')
        print()
        print()
        print()
        print()
