import os
from options import opt
import preprocess
from data import Data, data
import ner
import joint_train
import shutil
import test
import logging



logger = logging.getLogger()
if opt.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


logging.info(opt)

if opt.whattodo==1:

    train_token, train_entity, train_relation, train_name = preprocess.preprocess(data.train_dir)
    dev_token, dev_entity, dev_relation, dev_name = preprocess.preprocess(data.dev_dir)
    # test_token, test_entity, test_relation, test_name = preprocess.preprocess(data.test_dir)

    train_data = ner.generateData(train_token, train_entity, train_name)
    dev_data = ner.generateData(dev_token, dev_entity, dev_name)
    # test_data = ner.generateData(test_token, test_entity, test_name)

    if opt.pretrained_model_dir != 'None': # alphabet, label, etc. depend on pretrained model
        logging.info("load the config of pretrained model from {}".format(opt.pretrained_model_dir))
        old_data = Data()
        old_data.load(os.path.join(opt.pretrained_model_dir, 'data.pkl'))

        data.copy_alphabet(old_data)
        data.open_alphabet()

    else:
        old_data = None
        data.initial_feature_alphabets()

    logging.info("build ner alphabet ...")
    data.build_alphabet(train_data)
    if data.full_data:
        data.build_alphabet(dev_data)
        # data.build_alphabet(test_data)
    data.fix_alphabet()

    data.generate_instance('train', train_data)
    logging.info("generate_instance train completed")
    data.generate_instance('dev', dev_data)
    logging.info("generate_instance dev completed")
    # data.generate_instance('test', test_data)
    # print("generate_instance test completed")

    data.build_pretrain_emb()

    if opt.pretrained_model_dir != 'None':
        data.open_re_alphabet()
    else:
        data.initial_re_feature_alphabets()

    logging.info("build re alphabet ...")
    data.build_re_feature_alphabets(train_token, train_entity, train_relation)
    if data.full_data:
        data.build_re_feature_alphabets(dev_token, dev_entity, dev_relation)
        # data.build_re_feature_alphabets(test_token, test_entity, test_relation)
    data.fix_re_alphabet()

    # generate instance
    data.generate_re_instance('train', train_token, train_entity, train_relation, train_name)
    logging.info("generate_re_instance train completed")
    data.generate_re_instance('dev', dev_token, dev_entity, dev_relation, dev_name)
    logging.info("generate_re_instance dev completed")
    # data.generate_re_instance('test', test_token, test_entity, test_relation, test_name)
    # print("generate_re_instance test completed")

    data.show_data_summary()

    joint_train.joint_train(data, old_data, opt)

    data.clear_data()
    data.save(os.path.join(opt.output, 'data.pkl'))

else:

    if os.path.exists(opt.predict):
        shutil.rmtree(opt.predict)
        os.makedirs(opt.predict)
    else:
        os.makedirs(opt.predict)

    data.load(os.path.join(opt.output, "data.pkl"))

    data.MAX_SENTENCE_LENGTH = -1
    # for va begin
    data.nbest = 1
    data.sent_window = 1
    data.HP_gpu = opt.gpu
    # for va end
    data.show_data_summary()
    data.fix_alphabet()
    data.fix_re_alphabet()

    test.predict(opt, data)


