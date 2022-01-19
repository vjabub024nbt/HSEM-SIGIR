import torch
import torch.nn as nn
from kgdlg.modules.Attention import GlobalAttention, MultiHeadedAttention

from kgdlg.modules.StackedRNN import StackedGRU, StackedLSTM
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random


class DecoderBase(nn.Module):
    def forward(self, input, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        raise NotImplementedError


class AttnDecoder(DecoderBase):
    """ The GlobalAttention-based RNN decoder. """

    def __init__(self, rnn_type, embedding, attn_type, input_size,
                 hidden_size, num_layers=1, dropout=0.1, emo_embedding=None):
        super(AttnDecoder, self).__init__()

        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = embedding
        self.emo_embedding = emo_embedding
        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)

        if self.attn_type != 'none':
            self.attn = GlobalAttention(hidden_size, attn_type)

    def forward(self, input, context, state):
        if not self.emo_embedding is None:
            input, tgt_emo = input

            tgt_emo = tgt_emo.expand([input.size()[0], input.size()[1]])

            input_emb = self.embedding(input)
            emo_emb = self.emo_embedding(tgt_emo)

            emb = torch.cat([input_emb, emo_emb], 2)





        else:

            emb = self.embedding(input)

        emb = self.dropout(emb)

        state = state.expand(self.num_layers, state.size(1), state.size(2)).contiguous()
        rnn_outputs, hidden = self.rnn(emb, state)

        if self.attn_type != 'none':

            attn_outputs, attn_scores = self.attn(
                rnn_outputs.transpose(0, 1).contiguous(),
                context.transpose(0, 1)
            )

            outputs = attn_outputs
            attn = attn_outputs
        else:

            outputs = rnn_outputs
            attn = None

        return outputs, hidden, attn


class MHAttnDecoder(DecoderBase):
    """ The GlobalAttention-based RNN decoder. """

    def __init__(self, rnn_type, input_size,
                 hidden_size, num_layers=1, dropout=0.1, emo_embedding=None):
        super(MHAttnDecoder, self).__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)

        self.attn = MultiHeadedAttention(10, hidden_size)

    def forward(self, input, context, state):
        emb = input
        rnn_outputs, hidden = self.rnn(emb, state)

        attn_outputs = self.attn(
            rnn_outputs.transpose(0, 1).contiguous(),
            context.transpose(0, 1),
            context.transpose(0, 1)
        )

        outputs = self.dropout(attn_outputs)
        attn = attn_outputs

        return outputs, hidden, attn


class InputFeedDecoder(DecoderBase):
    def __init__(self, rnn_type, attn_type, embedding_size,
                 hidden_size, num_layers=1, dropout=0.1, emo_embedding=None):
        super(InputFeedDecoder, self).__init__()

        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        self.attn = GlobalAttention(hidden_size, attn_type)

    def forward(self, input, context, state):
        outputs = []
        attns = []
        output = self.init_input_feed(context).squeeze(0)
        emb = input
        hidden = state

        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output.contiguous(),
                context.transpose(0, 1)
            )

            output = self.dropout(attn_output)
            outputs += [output]
            attns += [attn]
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)

        return outputs, hidden, attns

    def init_decoder_state(self, enc_hidden):
        if not isinstance(enc_hidden, tuple):

            h = enc_hidden

        else:
            h = enc_hidden
        return h

    def init_input_feed(self, context):
        batch_size = context.size(1)
        hidden_size = self.hidden_size
        h_size = (batch_size, hidden_size)
        return context.data.new(*h_size).zero_().unsqueeze(0)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embedding_size + self.hidden_size

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):

        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        else:
            stacked_cell = StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)
