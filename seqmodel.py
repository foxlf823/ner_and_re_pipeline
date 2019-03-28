

import torch
import torch.nn as nn
from crf import CRF

class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()


        self.gpu = data.HP_gpu

        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        # data.label_alphabet_size += 2
        # self.word_hidden = WordSequence(data, False, True, data.use_char)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, label_size+2)


        self.crf = CRF(label_size, self.gpu)

        if torch.cuda.is_available():
            self.hidden2tag = self.hidden2tag.cuda(self.gpu)



    # def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        # outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, None, None)
    def neg_log_likelihood_loss(self, hidden, hidden_adv, batch_label, mask):
        if hidden_adv is not None:
            hidden = (hidden + hidden_adv)

        outs = self.hidden2tag(hidden)

        batch_size = hidden.size(0)


        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)


        total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, hidden, mask):

        outs = self.hidden2tag(hidden)


        scores, tag_seq = self.crf._viterbi_decode(outs, mask)

        return tag_seq


    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)


    def decode_nbest(self, hidden, mask, nbest):


        outs = self.hidden2tag(hidden)


        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

