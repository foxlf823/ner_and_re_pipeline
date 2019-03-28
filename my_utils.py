

from torch.utils.data import Dataset
import torch
import numpy as np
import re
import nltk
from data import data
from options import opt


pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2

def featureCapital(word):
    if word[0].isalpha() and word[0].isupper():
        return 1
    else:
        return 0



class RelationDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])



def sorted_collate1(batch):
    if len(batch[0]) ==2 : # I made a bad hypothesis
        x, y = zip(*batch)
    else:
        x = batch
        y = None

    x, y = pad1(x, y, data.pad_idx)

    if torch.cuda.is_available():
        for i, _ in enumerate(x):
            if x[i] is not None:
                if isinstance(x[i], list):
                    for j, _ in enumerate(x[i]):
                        x[i][j] = x[i][j].cuda(data.HP_gpu)
                else:
                    x[i] = x[i].cuda(data.HP_gpu)

        for i, _ in enumerate(y):
            if y[i] is not None:
                y[i] = y[i].cuda(data.HP_gpu)
    return x, y


def pad1(x, y, eos_idx):
    batch_size = len(x)

    words = [s['tokens'] for s in x]
    # features = [np.asarray(zip(s['cap'],s['postag'])) for s in x]
    features = [np.asarray(list(zip(s['postag']))) for s in x]
    feature_num = len(features[0][0])
    chars = [s['char'] for s in x]

    positions1 = [s['positions1'] for s in x]
    positions2 = [s['positions2'] for s in x]
    e1_type = [s['e1_type'] for s in x]
    e2_type = [s['e2_type'] for s in x]
    e1_token = [s['e1_token'] for s in x]
    e2_token = [s['e2_token'] for s in x]
    tok_num_betw = [s['tok_num_betw'] for s in x]
    et_num = [s['et_num'] for s in x]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()

    word_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    position1_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    position2_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len)).long())

    for idx, (seq, seq_pos1, seq_pos2, seqlen) in enumerate(list(zip(words, positions1, positions2, word_seq_lengths))):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        position1_seq_tensor[idx, :seqlen] = torch.LongTensor(seq_pos1)
        position2_seq_tensor[idx, :seqlen] = torch.LongTensor(seq_pos2)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    position1_seq_tensor = position1_seq_tensor[word_perm_idx]
    position2_seq_tensor = position2_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    pad_chars = [chars[idx] + [[0]] * (max_seq_len.item() - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(list(map(max, length_list)))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(list(zip(pad_chars, char_seq_lengths))):
        for idy, (word, wordlen) in enumerate(list(zip(seq, seqlen))):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len.item(), -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len.item(), )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)


    e1_length = [len(row) for row in e1_token]
    max_e1_length = max(e1_length)
    e1_token = pad_sequence(e1_token, max_e1_length, eos_idx)
    e1_length = torch.LongTensor(e1_length)
    e1_token = e1_token[word_perm_idx]
    e1_length = e1_length[word_perm_idx]

    e2_length = [len(row) for row in e2_token]
    max_e2_length = max(e2_length)
    e2_token = pad_sequence(e2_token, max_e2_length, eos_idx)
    e2_length = torch.LongTensor(e2_length)
    e2_token = e2_token[word_perm_idx]
    e2_length = e2_length[word_perm_idx]

    e1_type = torch.LongTensor(e1_type)
    e2_type = torch.LongTensor(e2_type)
    e1_type = e1_type[word_perm_idx]
    e2_type = e2_type[word_perm_idx]

    tok_num_betw = torch.LongTensor(tok_num_betw)
    tok_num_betw = tok_num_betw[word_perm_idx]

    et_num = torch.LongTensor(et_num)
    et_num = et_num[word_perm_idx]

    if y is not None:
        target = torch.LongTensor(y).view(-1)
        target_permute = torch.LongTensor(y)[torch.randperm(batch_size)]
        target = target[word_perm_idx]
        target_permute = target_permute[word_perm_idx]
    else:
        target = None
        target_permute = None

    return [word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths,
           char_seq_recover, position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type,
            e2_type, tok_num_betw, et_num], [target, target_permute]



def pad_sequence(x, max_len, eos_idx):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(eos_idx)
    for i, row in enumerate(x):
        assert eos_idx not in row, 'EOS in sequence {}'.format(row)
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x)

    return padded_x



def endless_get_next_batch_without_rebatch1(loaders, iters):
    try:
        x, y = next(iters)
    except StopIteration:
        iters = iter(loaders)
        x, y = next(iters)

    return x, y



def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True


def random_embedding(vocab_size, embedding_dim):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb

def reverse_grad(net):
    for p in net.parameters():
        if p.grad is not None: # word emb is not finetuned, so no grad
            p.grad *= -opt.lambd





