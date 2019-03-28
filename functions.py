

import numpy as np
import logging
import time

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(documents, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length):
    feature_num = len(feature_alphabets)

    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []

    for i, doc in enumerate(documents):
        start = time.time()
        for sentence in doc:
            for token in sentence:

                word = token['word']
                if number_normalized:
                    word = normalize_word(word)
                label = token['label']
                words.append(word)
                labels.append(label)
                word_Ids.append(word_alphabet.get_index(word))
                label_Ids.append(label_alphabet.get_index(label))
                ## get features
                feat_list = []
                feat_Id = []

                feat_idx = token['cap']
                feat_list.append(feat_idx)
                feat_Id.append(feature_alphabets[0].get_index(feat_idx))

                feat_idx = token['pos']
                feat_list.append(feat_idx)
                feat_Id.append(feature_alphabets[1].get_index(feat_idx))

                features.append(feat_list)
                feature_Ids.append(feat_Id)
                ## get char
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)

            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, features, chars, labels])
                instence_Ids.append([word_Ids, feature_Ids, char_Ids,label_Ids])
            words = []
            features = []
            chars = []
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []

        end = time.time()
        logging.debug("generate instance for %s-th doc finished. Time: %.2fs" % (i, end - start))

    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:

            pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:

            pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    logging.info("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim
       
def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def _readString(f):
    s = str()
    c = f.read(1).decode('iso-8859-1')
    while c != '\n' and c != ' ':
        s = s + c
        c = f.read(1).decode('iso-8859-1')

    return s

import struct
def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f))
            embedd_dim = int(_readString(f))

            for i in range(wordTotal):
                word = _readString(f)
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break
                try:
                    embedd_dict[word.decode('utf-8')] = word_vector
                except Exception as e:
                    pass
    else:
        with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                # feili
                if len(tokens) == 2:
                    continue # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0].decode('utf-8')] = embedd

    return embedd_dict, embedd_dim


