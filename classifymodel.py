
import torch
import torch.nn as nn
import torch.nn.functional as F
import my_utils

class ClassifyModel(nn.Module):
    def __init__(self, data):
        super(ClassifyModel, self).__init__()

        self.gpu = data.HP_gpu

        relation_alphabet_id = data.re_feature_name2id['[RELATION]']
        label_size = data.re_feature_alphabet_sizes[relation_alphabet_id]

        # self.word_hidden = WordSequence(data, True, False, False)

        self.attn = DotAttentionLayer(data.HP_hidden_dim, self.gpu)

        # instance-level feature
        entity_type_alphabet_id = data.re_feature_name2id['[ENTITY_TYPE]']
        self.entity_type_emb = nn.Embedding(data.re_feature_alphabets[entity_type_alphabet_id].size(),
                                       data.re_feature_emb_dims[entity_type_alphabet_id], data.pad_idx)
        self.entity_type_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[entity_type_alphabet_id].size(),
                                                           data.re_feature_emb_dims[entity_type_alphabet_id])))

        entity_alphabet_id = data.re_feature_name2id['[ENTITY]']
        self.entity_emb = nn.Embedding(data.re_feature_alphabets[entity_alphabet_id].size(),
                                       data.re_feature_emb_dims[entity_alphabet_id], data.pad_idx)
        self.entity_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[entity_alphabet_id].size(),
                                                           data.re_feature_emb_dims[entity_alphabet_id])))

        self.dot_att = DotAttentionLayer(data.re_feature_emb_dims[entity_alphabet_id], data.HP_gpu)

        tok_num_alphabet_id = data.re_feature_name2id['[TOKEN_NUM]']
        self.tok_num_betw_emb = nn.Embedding(data.re_feature_alphabets[tok_num_alphabet_id].size(),
                                       data.re_feature_emb_dims[tok_num_alphabet_id], data.pad_idx)
        self.tok_num_betw_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[tok_num_alphabet_id].size(),
                                                           data.re_feature_emb_dims[tok_num_alphabet_id])))

        et_num_alphabet_id = data.re_feature_name2id['[ENTITY_NUM]']
        self.et_num_emb = nn.Embedding(data.re_feature_alphabets[et_num_alphabet_id].size(),
                                       data.re_feature_emb_dims[et_num_alphabet_id], data.pad_idx)
        self.et_num_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[et_num_alphabet_id].size(),
                                                           data.re_feature_emb_dims[et_num_alphabet_id])))

        self.input_size = data.HP_hidden_dim + 2 * data.re_feature_emb_dims[entity_type_alphabet_id] + 2 * data.re_feature_emb_dims[entity_alphabet_id] + \
                          data.re_feature_emb_dims[tok_num_alphabet_id] + data.re_feature_emb_dims[et_num_alphabet_id]

        self.linear = nn.Linear(self.input_size, label_size, bias=False)

        self.loss_function = nn.NLLLoss(reduction='mean')

        if torch.cuda.is_available():
            self.attn = self.attn.cuda(data.HP_gpu)
            self.entity_type_emb = self.entity_type_emb.cuda(data.HP_gpu)
            self.entity_emb = self.entity_emb.cuda(data.HP_gpu)
            self.dot_att = self.dot_att.cuda(data.HP_gpu)
            self.tok_num_betw_emb = self.tok_num_betw_emb.cuda(data.HP_gpu)
            self.et_num_emb = self.et_num_emb.cuda(data.HP_gpu)
            self.linear = self.linear.cuda(data.HP_gpu)



    def neg_log_likelihood_loss(self, hidden, hidden_adv, word_seq_lengths, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num, targets):
        if hidden_adv is not None:
            hidden = (hidden + hidden_adv)

        hidden_features = self.attn((hidden, word_seq_lengths))

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        outs = self.linear(x)

        score = F.log_softmax(outs, 1)
        total_loss = self.loss_function(score, targets)
        _, tag_seq = torch.max(score, 1)
        return total_loss, tag_seq


    def forward(self, hidden, word_seq_lengths, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num):

        hidden_features = self.attn((hidden, word_seq_lengths))

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        outs = self.linear(x)


        _, tag_seq = torch.max(outs, 1)

        return tag_seq




class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size, gpu):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)
        self.gpu = gpu

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = F.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if torch.cuda.is_available():
            idxes = idxes.cuda(self.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output
