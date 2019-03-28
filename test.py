from os import listdir
from os.path import isfile, join
from preprocess import get_text_file, token_from_sent
import nltk
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

import pandas as pd
from seqmodel import SeqModel
from wordsequence import WordSequence
from classifymodel import ClassifyModel
import torch
import os
import my_utils
from functions import normalize_word
import ner
import bioc
import relation_extraction
import logging
import time

def generateDataForOneFile(doc_token):

    doc = []

    for sent_idx in range(9999): # this is an assumption, may be failed
        sent_token = doc_token[(doc_token['sent_idx'] == sent_idx)]

        if sent_token.shape[0] == 0:
            break

        sentence = []

        for _, token in sent_token.iterrows():
            word = token['text']
            pos = token['postag']
            cap = my_utils.featureCapital(word)

            token = {}
            token['word'] = word
            token['cap'] = cap
            token['pos'] = pos

            sentence.append(token)


        doc.append(sentence)

    return doc

def read_instance(doc, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length):
    feature_num = len(feature_alphabets)

    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []

    word_Ids = []
    feature_Ids = []
    char_Ids = []


    for sentence in doc:
        for token in sentence:

            # word = token['word'].decode('utf-8')
            word = token['word']
            if number_normalized:
                word = normalize_word(word)

            words.append(word)

            word_Ids.append(word_alphabet.get_index(word))

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
            instence_texts.append([words, features, chars])
            instence_Ids.append([word_Ids, feature_Ids, char_Ids])
        words = []
        features = []
        chars = []

        word_Ids = []
        feature_Ids = []
        char_Ids = []

    return instence_texts, instence_Ids

def evaluateWhenTest(data, wordseq, model, instence_Ids):

    instances = instence_Ids
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
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, mask= ner.batchify_without_label(instance, data.HP_gpu, True)
        hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, None, None)
        scores, nbest_tag_seq = model.decode_nbest(hidden, mask, data.nbest)
        nbest_pred_result = ner.recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
        nbest_pred_results += nbest_pred_result

    return nbest_pred_results

def predict(opt, data):

    seq_model = SeqModel(data)
    if opt.test_in_cpu:
        seq_model.load_state_dict(torch.load(os.path.join(opt.output, 'ner_model.pkl'), map_location='cpu'))
    else:
        cuda_src = 'cuda:{}'.format(opt.old_gpu)
        cuda_dst = 'cuda:{}'.format(opt.gpu)
        seq_model.load_state_dict(torch.load(os.path.join(opt.output, 'ner_model.pkl'), map_location={cuda_src:cuda_dst}))


    seq_wordseq = WordSequence(data, False, True, True, True)
    if opt.test_in_cpu:
        seq_wordseq.load_state_dict(torch.load(os.path.join(opt.output, 'ner_wordseq.pkl'), map_location='cpu'))
    else:
        cuda_src = 'cuda:{}'.format(opt.old_gpu)
        cuda_dst = 'cuda:{}'.format(opt.gpu)
        seq_wordseq.load_state_dict(torch.load(os.path.join(opt.output, 'ner_wordseq.pkl'), map_location={cuda_src:cuda_dst}))

    classify_model = ClassifyModel(data)
    if opt.test_in_cpu:
        classify_model.load_state_dict(torch.load(os.path.join(opt.output, 're_model.pkl'), map_location='cpu'))
    else:
        cuda_src = 'cuda:{}'.format(opt.old_gpu)
        cuda_dst = 'cuda:{}'.format(opt.gpu)
        classify_model.load_state_dict(torch.load(os.path.join(opt.output, 're_model.pkl'), map_location={cuda_src:cuda_dst}))

    classify_wordseq = WordSequence(data, True, False, True, False)
    if opt.test_in_cpu:
        classify_wordseq.load_state_dict(torch.load(os.path.join(opt.output, 're_wordseq.pkl'), map_location='cpu'))
    else:
        cuda_src = 'cuda:{}'.format(opt.old_gpu)
        cuda_dst = 'cuda:{}'.format(opt.gpu)
        classify_wordseq.load_state_dict(torch.load(os.path.join(opt.output, 're_wordseq.pkl'), map_location={cuda_src:cuda_dst}))

    input_files = [f for f in listdir(opt.input) if isfile(join(opt.input,f))]


    # for idx in tqdm(range(len(input_files))):
    for idx in range(len(input_files)):

        start = time.time()
        fileName = join(opt.input,input_files[idx])
        doc_name = input_files[idx]

        doc_token = processOneFile(fileName)

        doc = generateDataForOneFile(doc_token)

        raw_texts, raw_Ids = read_instance(doc, data.word_alphabet, data.char_alphabet,
                                                                   data.feature_alphabets, data.label_alphabet,
                                                                   data.number_normalized,
                                                                   data.MAX_SENTENCE_LENGTH)

        decode_results = evaluateWhenTest(data, seq_wordseq, seq_model, raw_Ids)


        entities = ner.translateNCRFPPintoEntities(doc_token, decode_results, doc_name)

        collection = bioc.BioCCollection()
        document = bioc.BioCDocument()
        collection.add_document(document)
        document.id = doc_name
        passage = bioc.BioCPassage()
        document.add_passage(passage)
        passage.offset = 0

        for entity in entities:
            anno_entity = bioc.BioCAnnotation()
            passage.add_annotation(anno_entity)
            anno_entity.id = entity.id
            anno_entity.infons['type'] = entity.type
            anno_entity_location = bioc.BioCLocation(entity.start, entity.getlength())
            anno_entity.add_location(anno_entity_location)
            anno_entity.text = entity.text


        test_X, test_other = relation_extraction.getRelationInstanceForOneDoc(doc_token, entities, doc_name, data)

        relations = relation_extraction.evaluateWhenTest(classify_wordseq, classify_model, test_X, data, test_other, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']])

        for relation in relations:
            bioc_relation = bioc.BioCRelation()
            passage.add_relation(bioc_relation)
            bioc_relation.id = relation.id
            bioc_relation.infons['type'] = relation.type

            node1 = bioc.BioCNode(relation.node1.id, 'argument 1')
            bioc_relation.add_node(node1)
            node2 = bioc.BioCNode(relation.node2.id, 'argument 2')
            bioc_relation.add_node(node2)


        with open(os.path.join(opt.predict, doc_name + ".bioc.xml"), 'w') as fp:
            bioc.dump(collection, fp)

        end = time.time()
        logging.info("process %s complete with %.2fs" % (input_files[idx], end-start))



    logging.info("test finished")



def processOneFile(fileName):

    corpus_file = get_text_file(fileName)

    # token
    all_sents_inds = []
    #generator = PunktSentenceTokenizer().span_tokenize(corpus_file)
    generator = sent_tokenizer.span_tokenize(corpus_file)
    for t in generator:
        all_sents_inds.append(t)

    df_doc = pd.DataFrame() # contains token-level information
    for ind in range(len(all_sents_inds)):
        t_start = all_sents_inds[ind][0]
        t_end = all_sents_inds[ind][1]
        tmp_tokens = token_from_sent(corpus_file[t_start:t_end], t_start)
        df_tokens = pd.DataFrame(tmp_tokens, columns=['text', 'postag', 'start', 'end'])

        df_sent_id = pd.DataFrame([ind]*len(tmp_tokens), columns = ['sent_idx'])
        df_comb = pd.concat([df_tokens, df_sent_id], axis=1)
        df_doc = pd.concat([df_doc, df_comb])

    df_doc.index = range(df_doc.shape[0])


    return df_doc
