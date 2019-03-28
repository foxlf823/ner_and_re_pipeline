
import torch

import numpy as np
import torch.autograd as autograd

import time
from metric import get_ner_fmeasure
import my_utils
from data_structure import Entity
from data import data
import logging



def getLabel(start, end, sent_entity):
    """
    Only considering the entity in ENTITY_TYPE. For double annotation, the first-meet entity is considered.
    :param start:
    :param end:
    :param sent_entity:
    :return:
    """
    match = ""
    for index, entity in sent_entity.iterrows():
        if start == entity['start'] and end == entity['end'] : # S
            match = "S"
            break
        elif start == entity['start'] and end != entity['end'] : # B
            match = "B"
            break
        elif start != entity['start'] and end == entity['end'] : # E
            match = "E"
            break
        elif start > entity['start'] and end < entity['end']:  # M
            match = "M"
            break

    if match != "":
        return match+"-"+sent_entity.loc[index]['type']
    else:
        return "O"

def generateData(tokens, entitys, names):
    documents = []
    for i in range(len(names)):
        start = time.time()
        doc_token = tokens[i]
        doc_entity = entitys[i]

        doc = []

        for sent_idx in range(9999): # this is an assumption, may be failed
            sent_token = doc_token[(doc_token['sent_idx'] == sent_idx)]
            sent_entity = doc_entity[(doc_entity['sent_idx'] == sent_idx)]

            if sent_token.shape[0] == 0:
                break

            sentence = []

            for _, token in sent_token.iterrows():
                word = token['text']
                pos = token['postag']
                cap = my_utils.featureCapital(word)
                label = getLabel(token['start'], token['end'], sent_entity)

                token = {}
                token['word'] = word
                token['cap'] = cap
                token['pos'] = pos
                token['label'] = label

                sentence.append(token)


            doc.append(sentence)

        documents.append(doc)
        end = time.time()
        logging.debug("generate data for %s finished. Time: %.2fs" % (names[i], end - start))

    return documents


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    with torch.no_grad():
        batch_size = len(input_batch_list)
        words = [sent[0] for sent in input_batch_list]
        features = [np.asarray(sent[1]) for sent in input_batch_list]
        feature_num = len(features[0][0])
        chars = [sent[2] for sent in input_batch_list]
        labels = [sent[3] for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        max_seq_len = word_seq_lengths.max()
        word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        permute_label_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
        feature_seq_tensors = []
        for idx in range(feature_num):
            feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len))).long())
        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
        for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            permute_label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)[torch.randperm(seqlen)]
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

        label_seq_tensor = label_seq_tensor[word_perm_idx]
        permute_label_seq_tensor = permute_label_seq_tensor[word_perm_idx]
        mask = mask[word_perm_idx]
        ### deal with char
        # pad_chars (batch_size, max_seq_len)
        pad_chars = [chars[idx] + [[0]] * (max_seq_len.item()-len(chars[idx])) for idx in range(len(chars))]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max(list(map(max, length_list)))
        char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                # print len(word), wordlen
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len.item(),-1)
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len.item(),)
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        if torch.cuda.is_available():
            word_seq_tensor = word_seq_tensor.cuda(data.HP_gpu)
            for idx in range(feature_num):
                feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda(data.HP_gpu)
            word_seq_lengths = word_seq_lengths.cuda(data.HP_gpu)
            word_seq_recover = word_seq_recover.cuda(data.HP_gpu)
            label_seq_tensor = label_seq_tensor.cuda(data.HP_gpu)
            permute_label_seq_tensor = permute_label_seq_tensor.cuda(data.HP_gpu)
            char_seq_tensor = char_seq_tensor.cuda(data.HP_gpu)
            char_seq_recover = char_seq_recover.cuda(data.HP_gpu)
            mask = mask.cuda(data.HP_gpu)
        return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask, permute_label_seq_tensor


def batchify_without_label(input_batch_list, gpu, volatile_flag=False):

    with torch.no_grad():
        batch_size = len(input_batch_list)
        words = [sent[0] for sent in input_batch_list]
        features = [np.asarray(sent[1]) for sent in input_batch_list]
        feature_num = len(features[0][0])
        chars = [sent[2] for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        max_seq_len = word_seq_lengths.max()
        word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()

        feature_seq_tensors = []
        for idx in range(feature_num):
            feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len))).long())
        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
        for idx, (seq, seqlen) in enumerate(list(zip(words, word_seq_lengths))):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

            mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

        mask = mask[word_perm_idx]
        ### deal with char
        # pad_chars (batch_size, max_seq_len)
        pad_chars = [chars[idx] + [[0]] * (max_seq_len.item()-len(chars[idx])) for idx in range(len(chars))]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max(list(map(max, length_list)))
        char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(list(zip(pad_chars, char_seq_lengths))):
            for idy, (word, wordlen) in enumerate(list(zip(seq, seqlen))):
                # print len(word), wordlen
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len.item(),-1)
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len.item(),)
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        if torch.cuda.is_available():
            word_seq_tensor = word_seq_tensor.cuda(data.HP_gpu)
            for idx in range(feature_num):
                feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda(data.HP_gpu)
            word_seq_lengths = word_seq_lengths.cuda(data.HP_gpu)
            word_seq_recover = word_seq_recover.cuda(data.HP_gpu)
            char_seq_tensor = char_seq_tensor.cuda(data.HP_gpu)
            char_seq_recover = char_seq_recover.cuda(data.HP_gpu)
            mask = mask.cuda(data.HP_gpu)
        return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, mask


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label




def evaluate(data, wordseq, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name")
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    wordseq.eval()
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, _ = batchify_with_label(instance, data.HP_gpu, True)
        if nbest:
            hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, None, None)
            scores, nbest_tag_seq = model.decode_nbest(hidden, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,batch_charrecover, None, None)
            tag_seq = model(hidden, mask)
        # print "tag:",tag_seq
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # print "word recover:", word_recover.size()
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    # print pred_variable.size()
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label

def evaluateWhenTest(data, wordseq, model):

    instances = data.raw_Ids
    nbest_pred_results = []
    wordseq.eval()
    model.eval()
    batch_size = data.HP_batch_size

    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, _  = batchify_with_label(instance, data.HP_gpu, True)
        hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, None, None)
        scores, nbest_tag_seq = model.decode_nbest(hidden, mask, data.nbest)
        nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
        nbest_pred_results += nbest_pred_result

    return nbest_pred_results



def checkWrongState(labelSequence):
    positionNew = -1
    positionOther = -1
    currentLabel = labelSequence[-1]
    assert currentLabel[0] == 'M' or currentLabel[0] == 'E'

    for j in range(len(labelSequence)-1)[::-1]:
        if positionNew == -1 and currentLabel[2:] == labelSequence[j][2:] and labelSequence[j][0] == 'B' :
            positionNew = j
        elif positionOther == -1 and (currentLabel[2:] != labelSequence[j][2:] or labelSequence[j][0] != 'M'):
            positionOther = j

        if positionOther != -1 and positionNew != -1:
            break

    if positionNew == -1:
        return False
    elif positionOther < positionNew:
        return True
    else:
        return False



def translateNCRFPPintoEntities(doc_token, predict_results, doc_name):

    entity_id = 1
    results = []

    sent_num = len(predict_results)
    for idx in range(sent_num):
        sent_length = len(predict_results[idx][0])
        sent_token = doc_token[(doc_token['sent_idx'] == idx)]

        assert sent_token.shape[0] == sent_length, "file {}, sent {}".format(doc_name, idx)
        labelSequence = []

        for idy in range(sent_length):
            token = sent_token.iloc[idy]
            label = predict_results[idx][0][idy]
            labelSequence.append(label)

            if label[0] == 'S' or label[0] == 'B':
                entity = Entity()
                entity.create(str(entity_id), label[2:], token['start'], token['end'], token['text'], idx, idy, idy)
                results.append(entity)
                entity_id += 1

            elif label[0] == 'M' or label[0] == 'E':
                if checkWrongState(labelSequence):
                    entity = results[-1]
                    entity.append(token['start'], token['end'], token['text'], idy)


    return results
