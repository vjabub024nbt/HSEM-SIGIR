from kgdlg.modules.Encoder import EncoderRNN
from kgdlg.modules.Decoder import AttnDecoder, InputFeedDecoder, MHAttnDecoder
from kgdlg.modules.Embedding import Embedding
from kgdlg.Model import NMTModel, MoSGenerator, CVAE, CvaeDialog, LatentNet, VariationalInference, DataCluster, \
    EmotionPreditor
import torch
import torch.nn as nn
import kgdlg.IO as IO


def create_emb_for_encoder_and_decoder(src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       padding_idx):
    embedding_encoder = Embedding(src_vocab_size, src_embed_size, padding_idx)
    embedding_decoder = Embedding(tgt_vocab_size, tgt_embed_size, padding_idx)

    return embedding_encoder, embedding_decoder


def create_encoder(opt, embedding):
    rnn_type = opt.rnn_type
    input_size = opt.embedding_size
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    dropout = opt.dropout
    bidirectional = opt.bidirectional

    encoder = EncoderRNN(rnn_type,
                         embedding,
                         input_size,
                         hidden_size,
                         num_layers,
                         dropout,
                         bidirectional)

    return encoder


def create_decoder(opt, embedding, emo_embedding):
    decoder_type = opt.decoder_type
    rnn_type = opt.rnn_type
    atten_model = opt.atten_model
    input_size = opt.embedding_size
    if opt.use_emo_emb:
        input_size += opt.emo_embedding_size

    print('decoder input size: ', input_size)

    if (opt.src_tgt_latent_merge_type in [0]) and (opt.vae_type in [6, 7, 8]):
        hidden_size = opt.hidden_size * 2
    elif (opt.use_src_or_tgt_attention in [3]) and (opt.vae_type in [6, 7, 8]):
        hidden_size = opt.hidden_size * 2
    else:
        hidden_size = opt.hidden_size

    num_layers = opt.num_layers
    dropout = opt.dropout

    if decoder_type == 'AttnDecoder':
        decoder = AttnDecoder(rnn_type,
                              embedding,
                              atten_model,
                              input_size,
                              hidden_size,
                              num_layers,
                              dropout,
                              emo_embedding)
    elif decoder_type == 'InputFeedDecoder':
        decoder = InputFeedDecoder(rnn_type,
                                   atten_model,
                                   input_size,
                                   hidden_size,
                                   num_layers,
                                   dropout,
                                   emo_embedding)

    elif decoder_type == 'MHAttnDecoder':
        decoder = MHAttnDecoder(rnn_type,
                                input_size,
                                hidden_size,
                                num_layers,
                                dropout,
                                emo_embedding)

    return decoder


def create_generator(input_size, output_size):
     
    generator = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LogSoftmax(dim=-1))
    return generator


def create_cvae(x_feat_size, c_feat_size, latent_size):
    net = CVAE(x_feat_size, c_feat_size, latent_size)
    return net


def create_gmm(opt):
    net = VariationalInference(opt)
    return net


def create_latent_net(opt):
    net = LatentNet(opt)
    return net


def create_data_cluster(opt):
    net = DataCluster(opt)
    return net


def create_emo_cls(opt):
    emo_num = len(opt.variable_emo_dict.itos)   
    emo_cls = nn.Linear(opt.embedding_size, emo_num)
    return emo_cls


def create_emo_predictor(src_emo_embedding, emo_pred):
    predictor = EmotionPreditor(src_emo_embedding, emo_pred)
    return predictor


def create_base_model(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    emo_vocab_size = len(fields['tgt_emo'].vocab)
    opt.variable_src_dict = fields["src"].vocab
    opt.variable_tgt_dict = fields["tgt"].vocab
    opt.variable_emo_dict = fields["tgt_emo"].vocab
    src_pad_idx = fields['src'].vocab.stoi[IO.PAD_WORD]
    src_unk_idx = fields['src'].vocab.stoi[IO.UNK_WORD]
    tgt_pad_idx = fields['tgt'].vocab.stoi[IO.PAD_WORD]
    tgt_unk_idx = fields['tgt'].vocab.stoi[IO.UNK_WORD]
     
     
     
     
     
     

     
    enc_embedding = Embedding(src_vocab_size, opt.embedding_size, src_pad_idx)
    dec_embedding = Embedding(tgt_vocab_size, opt.embedding_size, tgt_pad_idx)

    emo_embedding = None
    if opt.use_emo_emb:
        print('use_emo_emb')
        print('emo_vocab_size: ', emo_vocab_size)
        emo_embedding = nn.Embedding(emo_vocab_size, opt.emo_embedding_size)

     
    if opt.use_pretrained_word_vec:
        print('use_pretrained_word_vec')

         
        enc_pre_embeddings = fields['src'].vocab.vectors
        dec_pre_embeddings = fields['tgt'].vocab.vectors

        print(f'enc_pre_embeddings shape: {enc_pre_embeddings.shape}')
        print(f'dec_pre_embeddings shape: {dec_pre_embeddings.shape}')
        enc_embedding.embedding.weight.data.copy_(enc_pre_embeddings)
        dec_embedding.embedding.weight.data.copy_(dec_pre_embeddings)

         
        enc_embedding.embedding.weight.data[src_unk_idx] = torch.zeros(opt.embedding_size)
        enc_embedding.embedding.weight.data[src_pad_idx] = torch.zeros(opt.embedding_size)
        dec_embedding.embedding.weight.data[tgt_unk_idx] = torch.zeros(opt.embedding_size)
        dec_embedding.embedding.weight.data[tgt_pad_idx] = torch.zeros(opt.embedding_size)

     
    c_encoder = create_encoder(opt, enc_embedding)

    if opt.embedding_type in [1, 2]:
        x_encoder = create_encoder(opt, enc_embedding)
    elif opt.embedding_type in [3]:
        x_encoder = create_encoder(opt, dec_embedding)

    latent_net = create_latent_net(opt)
     
    decoder = create_decoder(opt, dec_embedding, emo_embedding)
    cvae_net = create_cvae(opt.hidden_size, opt.hidden_size, opt.latent_size)
    gmm_net = create_gmm(opt)
     
     
     
     
    ''' generator 用在每个 time step 把 hidden_size 维度的 dec_output 映射到 vocab_size 的单词输出概率'''
    generator = create_generator(opt.hidden_size, tgt_vocab_size)

    if opt.sample_rate_control:
        print('sample_rate_control')
        str2num = {'other': 0,
                   'like': 1,
                   'sadness': 2,
                   'disgust': 3,
                   'anger': 4,
                   'happiness': 5
                   }
        other = opt.variable_emo_dict.stoi['0']
        like = opt.variable_emo_dict.stoi['1']
        sad = opt.variable_emo_dict.stoi['2']
        disgust = opt.variable_emo_dict.stoi['3']
        anger = opt.variable_emo_dict.stoi['4']
        happy = opt.variable_emo_dict.stoi['5']
        sample_rate = [0] * 6
        sample_rate[other] = opt.other_rate
        sample_rate[like] = opt.like_rate
        sample_rate[sad] = opt.sad_rate
        sample_rate[disgust] = opt.disgust_rate
        sample_rate[anger] = opt.anger_rate
        sample_rate[happy] = opt.happy_rate
        opt.sample_rate = sample_rate
        print(opt.sample_rate)
    data_cluster = create_data_cluster(opt)
    emo_cls = create_emo_cls(opt)
     
     
     
     
     

     
    src_emo_embedding = nn.Embedding(emo_vocab_size, opt.src_emo_embedding_size)
    emo_pred = nn.Linear(opt.src_emo_embedding_size + opt.hidden_size, emo_vocab_size)
    emo_predictor = create_emo_predictor(src_emo_embedding, emo_pred)

    model = CvaeDialog(x_encoder, c_encoder, decoder, cvae_net, gmm_net, latent_net, generator, data_cluster, emo_cls,
                       opt, emo_predictor)
    return model
