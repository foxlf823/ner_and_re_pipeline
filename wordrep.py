
import torch
import torch.nn as nn
import numpy as np

from charcnn import CharCNN

class WordRep(nn.Module):
    def __init__(self, data, use_position, use_cap, use_postag, use_char):
        super(WordRep, self).__init__()

        self.gpu = data.HP_gpu
        self.use_char = use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0

        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim

            self.char_feature = CharCNN(data.char_alphabet.size(), None, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)

        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        self.feature_num = 0
        self.feature_embedding_dims = data.feature_emb_dims
        self.feature_embeddings = nn.ModuleList()

        if use_cap:
            self.feature_num += 1
            alphabet_id = data.feature_name2id['[Cap]']
            emb = nn.Embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])
            emb.weight.data.copy_(torch.from_numpy(
                self.random_embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])))
            self.feature_embeddings.append(emb)

        if use_postag:
            self.feature_num += 1
            alphabet_id = data.feature_name2id['[POS]']
            emb = nn.Embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])
            emb.weight.data.copy_(torch.from_numpy(
                self.random_embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])))
            self.feature_embeddings.append(emb)

        self.use_position = use_position
        if self.use_position:

            position_alphabet_id = data.re_feature_name2id['[POSITION]']
            self.position_embedding_dim = data.re_feature_emb_dims[position_alphabet_id]
            self.position1_emb = nn.Embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim, data.pad_idx)
            self.position1_emb.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim)))

            self.position2_emb = nn.Embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim, data.pad_idx)
            self.position2_emb.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim)))

        if torch.cuda.is_available():
            self.drop = self.drop.cuda(self.gpu)
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            for idx in range(self.feature_num):
                self.feature_embeddings[idx] = self.feature_embeddings[idx].cuda(self.gpu)
            if self.use_position:
                self.position1_emb = self.position1_emb.cuda(self.gpu)
                self.position2_emb = self.position2_emb.cuda(self.gpu)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, position1_inputs,
                position2_inputs):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embedding(word_inputs)
        word_list = [word_embs]
        for idx in range(self.feature_num):
            word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))

        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            ## concat word and char together
            word_list.append(char_features)



        if self.use_position:
            position1_feature = self.position1_emb(position1_inputs)
            position2_feature = self.position2_emb(position2_inputs)
            word_list.append(position1_feature)
            word_list.append(position2_feature)


        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent
        