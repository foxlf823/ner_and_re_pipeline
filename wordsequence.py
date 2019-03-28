
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from wordrep import WordRep

class WordSequence(nn.Module):
    def __init__(self, data, use_position, use_cap, use_postag, use_char):
        super(WordSequence, self).__init__()

        self.gpu = data.HP_gpu
        self.use_char = use_char
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = True
        self.lstm_layer = 1
        self.wordrep = WordRep(data, use_position, use_cap, use_postag, use_char)
        self.tune_wordemb = data.tune_wordemb

        self.input_size = data.word_emb_dim
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim


        if use_cap:
            self.input_size += data.feature_emb_dims[data.feature_name2id['[Cap]']]
        if use_postag:
            self.input_size += data.feature_emb_dims[data.feature_name2id['[POS]']]

        self.use_position = use_position
        if self.use_position:
            self.input_size += 2*data.re_feature_emb_dims[data.re_feature_name2id['[POSITION]']]

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim


        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)


        if torch.cuda.is_available():
            self.droplstm = self.droplstm.cuda(self.gpu)

            self.lstm = self.lstm.cuda(self.gpu)



    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                position1_inputs, position2_inputs):

        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                                      position1_inputs, position2_inputs)
        ## word_embs (batch_size, seq_len, embed_size)

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)
        return feature_out

