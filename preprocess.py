
from os import listdir
from os.path import isfile, join
import pandas as pd
import bioc
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
import logging
import my_utils
import time
import codecs

def preprocess(basedir):
    annotation_dir = join(basedir, 'annotations')
    corpus_dir = join(basedir, 'corpus')

    annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir,f)) and f[0]!='.']
    # corpus_files = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir,f))]

    df_all_doc = []
    df_all_entity = []
    df_all_relation = []
    all_name = []

    for idx in range(len(annotation_files)):
        start = time.time()
        fileName = annotation_files[idx]

        df_doc, df_entity, df_relation = processOneFile(fileName, annotation_dir, corpus_dir)
        fileName = fileName[0:fileName.find('.')]
        df_all_doc.append(df_doc)
        df_all_entity.append(df_entity)
        df_all_relation.append(df_relation)
        all_name.append(fileName)
        end = time.time()
        logging.debug("preprocess %s finished. Time: %.2fs" % (fileName, end - start))

    logging.info("preprocessing complete in {}".format(basedir))

    return df_all_doc, df_all_entity, df_all_relation, all_name



def processOneFile(fileName, annotation_dir, corpus_dir):
    annotation_file = get_bioc_file(join(annotation_dir, fileName))
    corpus_file = get_text_file(join(corpus_dir, fileName.split('.bioc')[0]))
    bioc_passage = annotation_file[0].passages[0]

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

    # entity
    entity_span_in_this_passage = [] # to determine entity overlap
    anno_data = []
    for entity in bioc_passage.annotations:

        start = entity.locations[0].offset
        end = entity.locations[0].end

        tmp_df = pd.DataFrame(entity_span_in_this_passage, columns = ['start','end', 'type'])
        #result_df = tmp_df[((tmp_df['start']<=start)&(tmp_df['end']>start)) | ((tmp_df['start']<end)&(tmp_df['end']>=end))]
        result_df = tmp_df[
            ((tmp_df['start'] <= start) & (tmp_df['end'] > start)) |
            ((tmp_df['start'] < end) & (tmp_df['end'] >= end)) |
            ((tmp_df['start'] >= start) & (tmp_df['end'] <= end)) |
            ((tmp_df['start'] <= start) & (tmp_df['end'] >= end))
        ]

        if result_df.shape[0]==0:
            sent_idx = entity_in_sentence(start, end, all_sents_inds)
            if sent_idx == -1:
                # logging.debug('file {}, entity {}, cannot find entity in all sentences'.format(fileName, entity.id))
                continue

            df_sentence = df_doc[(df_doc['sent_idx'] == sent_idx)]
            tf_start = -1
            tf_end = -1
            token_num = df_sentence.shape[0]

            for tf_idx in range(token_num):
                token = df_sentence.iloc[tf_idx]

                if token['start'] == start:
                    tf_start = tf_idx
                if token['end'] == end:
                    tf_end = tf_idx

            if tf_start == -1 or tf_end == -1:  # due to tokenization error, e.g., 10_197, hyper-CVAD-based vs hyper-CVAD
                # logging.debug('file {}, entity {}, not found tf_start or tf_end'.format(fileName, entity.id))
                continue

            anno_data.append([entity.id, start, end, entity.text, entity.infons['type'], sent_idx, tf_start, tf_end])
            entity_span_in_this_passage.append([start, end, entity.infons['type']])
        else: # some entities overlap with current entity
            total_overlap = False
            for _, overlap in result_df.iterrows():
                if overlap['start'] == start and overlap['end'] == end:
                    total_overlap = True
                    break

            if total_overlap:
                # logging.debug('file {}, entity {}, double annotation'.format(fileName, entity.id))
                # double annotation
                sent_idx = entity_in_sentence(start, end, all_sents_inds)
                if sent_idx == -1:
                    raise RuntimeError('file {}, entity {}, cannot find entity in all sentences'.format(fileName, entity.id))
                    continue

                df_sentence = df_doc[(df_doc['sent_idx'] == sent_idx)]
                tf_start = -1
                tf_end = -1
                token_num = df_sentence.shape[0]

                for tf_idx in range(token_num):
                    token = df_sentence.iloc[tf_idx]

                    if token['start'] == start:
                        tf_start = tf_idx
                    if token['end'] == end:
                        tf_end = tf_idx

                if tf_start == -1 or tf_end == -1:
                    raise RuntimeError('file {}, entity {}, not found tf_start or tf_end'.format(fileName, entity.id))
                    continue

                anno_data.append(
                    [entity.id, start, end, entity.text, entity.infons['type'], sent_idx, tf_start, tf_end])
                entity_span_in_this_passage.append([start, end, entity.infons['type']])

            # else:
                # logging.debug('file {}, entity {}, overlapped'.format(fileName, entity.id))




    df_entity = pd.DataFrame(anno_data, columns = ['id','start','end','text','type', 'sent_idx', 'tf_start', 'tf_end']) # contains entity information
    df_entity = df_entity.sort_values('start')
    df_entity.index = range(df_entity.shape[0])

    # relation
    relation_data = []
    for relation in bioc_passage.relations:
        argument1 = relation.nodes[0].refid
        argument2 = relation.nodes[1].refid
        re_type = relation.infons['type']

        df_result = df_entity[ ((df_entity['id']==argument1) | (df_entity['id']==argument2)) ]
        if df_result.shape[0]==2:
            relation_data.append([relation.id, re_type, argument1, argument2])
        else:
            # logging.debug("file {}, relation {}, argument can't be found".format(fileName, relation.id))
            continue

    df_relation = pd.DataFrame(relation_data, columns = ['id','type','entity1_id','entity2_id'])

    return df_doc, df_entity, df_relation


def get_bioc_file(filename):
    list_result = []
    with bioc.iterparse(filename) as parser:
        for document in parser:
            list_result.append(document)
    return list_result

def get_text_file(filename):
    # return open(filename,'r').read()
    with codecs.open(filename, 'r', 'UTF-8') as fp:
        return fp.read()

def text_tokenize(txt, sent_start):

    tokens=my_utils.my_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset+sent_start, offset+len(token)+sent_start
        offset += len(token)

def text_tokenize_and_postagging(txt, sent_start):
    tokens=my_utils.my_tokenize(txt)
    pos_tags = nltk.pos_tag(tokens)

    offset = 0
    for token, pos_tag in pos_tags:
        offset = txt.find(token, offset)
        yield token, pos_tag, offset+sent_start, offset+len(token)+sent_start
        offset += len(token)

def token_from_sent(txt, sent_start):
    #return [token for token in text_tokenize(txt, sent_start)]
    return [token for token in text_tokenize_and_postagging(txt, sent_start)]

def entity_in_sentence(entity_start, entity_end, all_sents_inds):
    for i, (start, end) in enumerate(all_sents_inds):
        if entity_start >= start and entity_end <= end:
            return i
    # due to annotation or sentence splitter error
    return -1


