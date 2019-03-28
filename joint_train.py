from seqmodel import SeqModel
from wordsequence import WordSequence
from classifymodel import ClassifyModel
import torch
import itertools
import torch.optim as optim
import my_utils
from torch.utils.data import DataLoader
import time
import random
from ner import batchify_with_label

import ner
import os
import relation_extraction
import logging



def makeRelationDataset(re_X_positive, re_Y_positive, re_X_negative, re_Y_negative, ratio, b_shuffle, my_collate, batch_size):

    a = list(range(len(re_X_negative)))
    random.shuffle(a)
    indices = a[:int(len(re_X_negative)*ratio)]

    temp_X = []
    temp_Y = []
    for i in range(len(re_X_positive)):
        temp_X.append(re_X_positive[i])
        temp_Y.append(re_Y_positive[i])
    for i in range(len(indices)):
        temp_X.append(re_X_negative[indices[i]])
        temp_Y.append(re_Y_negative[indices[i]])

    data_set = my_utils.RelationDataset(temp_X, temp_Y)

    data_loader = DataLoader(data_set, batch_size, shuffle=b_shuffle, collate_fn=my_collate)
    it = iter(data_loader)
    return data_loader, it


def joint_train(data, old_data, opt):

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    if opt.pretrained_model_dir != 'None':
        seq_model = SeqModel(data)
        if opt.test_in_cpu:
            seq_model.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 'ner_model.pkl'), map_location='cpu'))
        else:
            seq_model.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 'ner_model.pkl')))

        if (data.label_alphabet_size != seq_model.crf.tagset_size)\
                or (data.HP_hidden_dim != seq_model.hidden2tag.weight.size(1)):
            raise RuntimeError("ner_model not compatible")

        seq_wordseq = WordSequence(data, False, True, True, True)

        if ((data.word_emb_dim != seq_wordseq.wordrep.word_embedding.embedding_dim)\
            or (data.char_emb_dim != seq_wordseq.wordrep.char_feature.char_embeddings.embedding_dim)\
            or (data.feature_emb_dims[0] != seq_wordseq.wordrep.feature_embedding_dims[0])\
            or (data.feature_emb_dims[1] != seq_wordseq.wordrep.feature_embedding_dims[1])):
            raise RuntimeError("ner_wordseq not compatible")

        old_seq_wordseq = WordSequence(old_data, False, True, True, True)
        if opt.test_in_cpu:
            old_seq_wordseq.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 'ner_wordseq.pkl'), map_location='cpu'))
        else:
            old_seq_wordseq.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 'ner_wordseq.pkl')))

        # sd = old_seq_wordseq.lstm.state_dict()

        for old, new in zip(old_seq_wordseq.lstm.parameters(), seq_wordseq.lstm.parameters()):
            new.data.copy_(old)

        vocab_size = old_seq_wordseq.wordrep.word_embedding.num_embeddings
        seq_wordseq.wordrep.word_embedding.weight.data[0:vocab_size, :] = old_seq_wordseq.wordrep.word_embedding.weight.data[0:vocab_size,:]

        vocab_size = old_seq_wordseq.wordrep.char_feature.char_embeddings.num_embeddings
        seq_wordseq.wordrep.char_feature.char_embeddings.weight.data[0:vocab_size, :] = old_seq_wordseq.wordrep.char_feature.char_embeddings.weight.data[0:vocab_size, :]

        for i, feature_embedding in enumerate(old_seq_wordseq.wordrep.feature_embeddings):
            vocab_size = feature_embedding.num_embeddings
            seq_wordseq.wordrep.feature_embeddings[i].weight.data[0:vocab_size, :] = feature_embedding.weight.data[0:vocab_size, :]

        # for word in data.word_alphabet.iteritems():
        #
        #     old_seq_wordseq.wordrep.word_embedding.weight.data data.word_alphabet.get_index(word)
        #

        classify_wordseq = WordSequence(data, True, False, True, False)

        if ((data.word_emb_dim != classify_wordseq.wordrep.word_embedding.embedding_dim)\
            or (data.re_feature_emb_dims[data.re_feature_name2id['[POSITION]']] != classify_wordseq.wordrep.position1_emb.embedding_dim)\
            or (data.feature_emb_dims[1] != classify_wordseq.wordrep.feature_embedding_dims[0])):
            raise RuntimeError("re_wordseq not compatible")

        old_classify_wordseq = WordSequence(old_data, True, False, True, False)
        if opt.test_in_cpu:
            old_classify_wordseq.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 're_wordseq.pkl'), map_location='cpu'))
        else:
            old_classify_wordseq.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 're_wordseq.pkl')))


        for old, new in zip(old_classify_wordseq.lstm.parameters(), classify_wordseq.lstm.parameters()):
            new.data.copy_(old)
            
        vocab_size = old_classify_wordseq.wordrep.word_embedding.num_embeddings
        classify_wordseq.wordrep.word_embedding.weight.data[0:vocab_size, :] = old_classify_wordseq.wordrep.word_embedding.weight.data[0:vocab_size,:]

        vocab_size = old_classify_wordseq.wordrep.position1_emb.num_embeddings
        classify_wordseq.wordrep.position1_emb.weight.data[0:vocab_size, :] = old_classify_wordseq.wordrep.position1_emb.weight.data[0:vocab_size, :]

        vocab_size = old_classify_wordseq.wordrep.position2_emb.num_embeddings
        classify_wordseq.wordrep.position2_emb.weight.data[0:vocab_size, :] = old_classify_wordseq.wordrep.position2_emb.weight.data[0:vocab_size, :]

        vocab_size = old_classify_wordseq.wordrep.feature_embeddings[0].num_embeddings
        classify_wordseq.wordrep.feature_embeddings[0].weight.data[0:vocab_size, :] = old_classify_wordseq.wordrep.feature_embeddings[0].weight.data[0:vocab_size, :]


        classify_model = ClassifyModel(data)

        old_classify_model = ClassifyModel(old_data)
        if opt.test_in_cpu:
            old_classify_model.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 're_model.pkl'), map_location='cpu'))
        else:
            old_classify_model.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 're_model.pkl')))

        if (data.re_feature_alphabet_sizes[data.re_feature_name2id['[RELATION]']] != old_classify_model.linear.weight.size(0)
            or (data.re_feature_emb_dims[data.re_feature_name2id['[ENTITY_TYPE]']] != old_classify_model.entity_type_emb.embedding_dim) \
                or (data.re_feature_emb_dims[
                        data.re_feature_name2id['[ENTITY]']] != old_classify_model.entity_emb.embedding_dim)
                or (data.re_feature_emb_dims[
                        data.re_feature_name2id['[TOKEN_NUM]']] != old_classify_model.tok_num_betw_emb.embedding_dim) \
                or (data.re_feature_emb_dims[
                        data.re_feature_name2id['[ENTITY_NUM]']] != old_classify_model.et_num_emb.embedding_dim) \
                ):
            raise RuntimeError("re_model not compatible")

        vocab_size = old_classify_model.entity_type_emb.num_embeddings
        classify_model.entity_type_emb.weight.data[0:vocab_size, :] = old_classify_model.entity_type_emb.weight.data[0:vocab_size, :]

        vocab_size = old_classify_model.entity_emb.num_embeddings
        classify_model.entity_emb.weight.data[0:vocab_size, :] = old_classify_model.entity_emb.weight.data[0:vocab_size, :]

        vocab_size = old_classify_model.tok_num_betw_emb.num_embeddings
        classify_model.tok_num_betw_emb.weight.data[0:vocab_size, :] = old_classify_model.tok_num_betw_emb.weight.data[0:vocab_size, :]

        vocab_size = old_classify_model.et_num_emb.num_embeddings
        classify_model.et_num_emb.weight.data[0:vocab_size, :] = old_classify_model.et_num_emb.weight.data[0:vocab_size, :]


    else:
        seq_model = SeqModel(data)
        seq_wordseq = WordSequence(data, False, True, True, True)

        classify_wordseq = WordSequence(data, True, False, True, False)
        classify_model = ClassifyModel(data)


    iter_parameter = itertools.chain(*map(list, [seq_wordseq.parameters(), seq_model.parameters()]))
    seq_optimizer = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)
    iter_parameter = itertools.chain(*map(list, [classify_wordseq.parameters(), classify_model.parameters()]))
    classify_optimizer = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)

    if data.tune_wordemb == False:
        my_utils.freeze_net(seq_wordseq.wordrep.word_embedding)
        my_utils.freeze_net(classify_wordseq.wordrep.word_embedding)

    re_X_positive = []
    re_Y_positive = []
    re_X_negative = []
    re_Y_negative = []
    relation_vocab = data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']]
    my_collate = my_utils.sorted_collate1
    for i in range(len(data.re_train_X)):
        x = data.re_train_X[i]
        y = data.re_train_Y[i]

        if y != relation_vocab.get_index("</unk>"):
            re_X_positive.append(x)
            re_Y_positive.append(y)
        else:
            re_X_negative.append(x)
            re_Y_negative.append(y)

    re_dev_loader = DataLoader(my_utils.RelationDataset(data.re_dev_X, data.re_dev_Y), data.HP_batch_size, shuffle=False, collate_fn=my_collate)
    # re_test_loader = DataLoader(my_utils.RelationDataset(data.re_test_X, data.re_test_Y), data.HP_batch_size, shuffle=False, collate_fn=my_collate)

    best_ner_score = -1
    best_re_score = -1
    count_performance_not_grow = 0

    for idx in range(data.HP_iteration):
        epoch_start = time.time()

        seq_wordseq.train()
        seq_wordseq.zero_grad()
        seq_model.train()
        seq_model.zero_grad()

        classify_wordseq.train()
        classify_wordseq.zero_grad()
        classify_model.train()
        classify_model.zero_grad()

        batch_size = data.HP_batch_size

        random.shuffle(data.train_Ids)
        ner_train_num = len(data.train_Ids)
        ner_total_batch = ner_train_num // batch_size + 1

        re_train_loader, re_train_iter = makeRelationDataset(re_X_positive, re_Y_positive, re_X_negative, re_Y_negative,
                                                             data.unk_ratio, True, my_collate, data.HP_batch_size)
        re_total_batch = len(re_train_loader)

        total_batch = max(ner_total_batch, re_total_batch)
        min_batch = min(ner_total_batch, re_total_batch)

        for batch_id in range(total_batch):


            if batch_id < ner_total_batch:
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > ner_train_num:
                    end = ner_train_num
                instance = data.train_Ids[start:end]
                batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, \
                    batch_permute_label = batchify_with_label(instance, data.HP_gpu)


                hidden = seq_wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                             batch_charrecover, None, None)
                hidden_adv = None
                loss, tag_seq = seq_model.neg_log_likelihood_loss(hidden, hidden_adv, batch_label, mask)
                loss.backward()
                seq_optimizer.step()
                seq_wordseq.zero_grad()
                seq_model.zero_grad()


            if batch_id < re_total_batch:
                [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
                 batch_char, batch_charlen, batch_charrecover, \
                 position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
                 tok_num_betw, et_num], [targets, targets_permute] = my_utils.endless_get_next_batch_without_rebatch1(
                    re_train_loader, re_train_iter)

                hidden = classify_wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                                  batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                hidden_adv = None
                loss, pred = classify_model.neg_log_likelihood_loss(hidden, hidden_adv, batch_wordlen,
                                                                    e1_token, e1_length, e2_token, e2_length, e1_type,
                                                                    e2_type,
                                                                    tok_num_betw, et_num, targets)
                loss.backward()
                classify_optimizer.step()
                classify_wordseq.zero_grad()
                classify_model.zero_grad()


        epoch_finish = time.time()
        logging.info("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))

        _, _, _, _, ner_score, _, _ = ner.evaluate(data, seq_wordseq, seq_model, "dev")
        logging.info("ner evaluate: f: %.4f" % (ner_score))
        if ner_score > best_ner_score:
            logging.info("new best score: ner: %.4f" % (ner_score))
            best_ner_score = ner_score

            torch.save(seq_wordseq.state_dict(), os.path.join(opt.output, 'ner_wordseq.pkl'))
            torch.save(seq_model.state_dict(), os.path.join(opt.output, 'ner_model.pkl'))

            count_performance_not_grow = 0

            # _, _, _, _, test_ner_score, _, _ = ner.evaluate(data, seq_wordseq, seq_model, "test")
            # logging.info("ner evaluate on test: f: %.4f" % (test_ner_score))

        else:
            count_performance_not_grow += 1


        re_score = relation_extraction.evaluate(classify_wordseq, classify_model, re_dev_loader)
        logging.info("re evaluate: f: %.4f" % (re_score))
        if re_score > best_re_score:
            logging.info("new best score: re: %.4f" % (re_score))
            best_re_score = re_score

            torch.save(classify_wordseq.state_dict(), os.path.join(opt.output, 're_wordseq.pkl'))
            torch.save(classify_model.state_dict(), os.path.join(opt.output, 're_model.pkl'))

            count_performance_not_grow = 0

            # test_re_score = relation_extraction.evaluate(classify_wordseq, classify_model, re_test_loader)
            # logging.info("re evaluate on test: f: %.4f" % (test_re_score))
        else:
            count_performance_not_grow += 1


        if count_performance_not_grow > 2*data.patience:
            logging.info("early stop")
            break

    logging.info("train finished")