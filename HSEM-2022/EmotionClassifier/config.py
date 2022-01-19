"""
hyper parameters
"""
mark = 'bert'
SEED = 1
gpuid = 3

# data
batch_size = 64
vocab_size = 50000
train_rate = 0.9
valid_rate = 0.05
test_rate = 0.05
data_path = 'data'
train_file = 'train_3.txt'
valid_file = 'dev_3.txt'
test_file = 'test_3.txt'
# text_field = 'text_field_3.pkl'
# label_field = 'label_field_3.pkl'  # defaultdict(None, {'0': 0, '1': 1, '3': 2, '5': 3, '2': 4, '4': 5})
text_field = 'text_field.pkl'
label_field = 'label_field.pkl'
custom_word_embedding = 'sgns.weibo.bigram-char'
# custom_word_embedding = 'sgns.weibo.word'
best_model = 'bert_blstm_64.52.pkl'

# model
# bidirectional = True
# embedding_dim = 300
# hidden_dim = 512
# n_layers = 3
# dropout = 0.6
# lr = 1e-3  # learning rate
# weight_decay = 1e-4
# epochs = 50
use_bert = True
bert_pretrained_name = 'bert-base-chinese'

if not use_bert:
    bidirectional = True
    embedding_dim = 300
    hidden_dim = 512
    n_layers = 3
    dropout = 0.7
    lr = 1e-3  # learning rate
    weight_decay = 1e-4
    epochs = 50

else:
    bidirectional = True
    hidden_dim = 600
    n_layers = 2
    dropout = 0.5
    lr = 5e-5  # learning rate
    weight_decay = 5e-4
    epochs = 50
