import math
import os
import argparse
import sys
import numpy as np
import tensorflow as tf
import time

sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from src.utils import pre_logger
from src.configuration import ChatConfig
from data_utils.prepare_dialogue_data import get_word_count, read_emotion_words, construct_vocab, construct_word_dict
from data_utils.prepare_dialogue_data import read_training_file, align_sentence_length, get_predict_train_response_data
from data_utils.prepare_dialogue_data import read_emotion_label, align_batch_size, shuffle_train_data, get_word_list
from data_utils.prepare_dialogue_data import read_word_embeddings, filter_test_sentence_length, write_test_data
from data_utils.prepare_dialogue_data import filter_sentence_length, read_stop_words, read_total_embeddings
from data_utils.prepare_dialogue_data import align_test_batch_size
from src.model import EmotionChatMachine


FLAGS = None


def train(config_file, pre_train_word_count_file, emotion_words_dir, post_file, response_file, emotion_label_file,
          embedding_file, train_word_count, session, checkpoint_dir, max_vocab_size, test_post_file, test_response_file,
          test_label_file, valid_result_file, restore_model):
    """
    train the dialogue model
    :param config_file:
    :param pre_train_word_count_file:
    :param emotion_words_dir:
    :param post_file:
    :param response_file:
    :param emotion_label_file:
    :param embedding_file:
    :param train_word_count:
    :param session:
    :param checkpoint_dir:
    :param max_vocab_size:
    :param test_post_file:
    :param test_label_file:
    :return:
    """
    log_name = "dialogue"
    logger = pre_logger(log_name)

    chat_config = ChatConfig(config_file)

    logger.info("Now prepare data!\n")
    logger.info("Read stop words!\n")
    # stop_words = read_stop_words(FLAGS.stop_words_file)

    logger.info("Construct vocab first\n")
    total_embeddings, total_word2id, total_word_list = read_total_embeddings(embedding_file, max_vocab_size)
    pre_word_count = get_word_count(pre_train_word_count_file, chat_config.word_count)
    emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,
                                chat_config.emotion_vocab_size, FLAGS.unk)
    word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    id2words = {idx: word for word, idx in word_dict.items()}
    word_unk_id = word_dict[FLAGS.unk]
    assert word_unk_id == word_dict['</s>']
    word_start_id = word_dict[FLAGS.start_symbol]
    word_end_id = word_dict[FLAGS.end_symbol]
    final_word_list = get_word_list(id2words)

    logger.info("Read word embeddings!\n")
    embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)
    print(np.array(embeddings).shape)

    logger.info("Read test data\n")
    test_post_data = read_training_file(test_post_file, word_dict, FLAGS.unk)
    test_response_data = read_training_file(test_response_file, word_dict, FLAGS.unk)
    test_label_data = read_emotion_label(test_label_file)
    test_length = len(test_post_data)
    print('test_length: ', test_length)
    test_post_data_length = [len(post_data) for post_data in test_post_data]

    logger.info("Align sentence length by padding!\n")
    test_post_data = align_sentence_length(test_post_data, chat_config.max_len, word_unk_id)
    test_post_data, test_post_data_length, test_label_data = \
        align_test_batch_size(test_post_data, test_post_data_length, test_label_data, chat_config.batch_size)

    logger.info("Finish preparing test data!\n")

    # vliad data
    logger.info("Read valid data\n")
    valid_post_data = read_training_file(test_post_file, word_dict, FLAGS.unk)
    valid_response_data = read_training_file(test_response_file, word_dict, FLAGS.unk)
    valid_label_data = read_emotion_label(test_label_file)
    valid_post_length = [len(post_data) for post_data in valid_post_data]
    valid_response_length = [len(post_data) for post_data in valid_post_data]

    logger.info("Align sentence length by padding!\n")
    valid_post_data = align_sentence_length(valid_post_data, chat_config.max_len, word_unk_id)
    valid_response_data, valid_predict_response_data = get_predict_train_response_data(valid_response_data,
                                                                                       word_start_id,
                                                                                       word_end_id, word_unk_id,
                                                                                       chat_config.max_len)
    valid_post_data, valid_post_length, valid_response_data, valid_predict_response_data, valid_emotion_labels = \
        align_batch_size(valid_post_data, valid_post_length, valid_response_data, valid_predict_response_data,
                         valid_label_data,
                         chat_config.batch_size)

    logger.info("Finish preparing valid data!\n")

    logger.info("Define model\n")
    class_weights = np.ones([len(word_dict)])
    class_weights[word_unk_id] = 0
    emotion_chat_machine = EmotionChatMachine(config_file, session, word_dict, class_weights, embeddings,
                                              chat_config.generic_word_size + 3, word_start_id, word_end_id,
                                              "emotion_chat_machine")
    valid_result_out = open(valid_result_file, 'w', encoding='utf8')

    checkpoint_path = os.path.join(checkpoint_dir, f'dialogue-model-{restore_model}')
    emotion_chat_machine.saver.restore(session, checkpoint_path)

    logger.info("Generate test data!\n")
    start_time = time.time()
    test_batch = int(len(test_post_data) / chat_config.batch_size)
    generate_data = []
    for k in range(test_batch):
        this_post_data, this_post_len, this_emotion_labels, this_emotion_mask = \
            emotion_chat_machine.get_test_batch(test_post_data, test_post_data_length, test_label_data, k)
        generate_words, scores, new_embeddings = emotion_chat_machine.generate_step(this_post_data, this_post_len,
                                                                                    this_emotion_labels,
                                                                                    this_emotion_mask)
        generate_data.extend(generate_words)

    generate_data = generate_data[: test_length]
    test_label_data = test_label_data[: test_length]

    write_test_data(generate_data, FLAGS.generate_response_file + f'.epoch_{restore_model}.', id2words, test_label_data)
    test_time = time.time() - start_time
    print(f'test_time: {test_time}')
    print(f'{test_time / test_batch}s/per batch*sentence')
    # logger.info("Valid data!\n")
    # total_entropy_loss=0
    # total_words=0
    # num_valid_batch = int(len(valid_post_data) /chat_config.batch_size)
    # for j in range(num_valid_batch):
    #     this_post_data, this_post_len, this_train_res_data, this_predict_res_data, this_emotion_labels, \
    #     this_emotion_mask = emotion_chat_machine.get_batch(valid_post_data, valid_post_length, valid_response_data,
    #                                                    valid_predict_response_data, valid_emotion_labels,j)
    #
    #     n_words = 0
    #     for batch in this_predict_res_data:
    #         for word in batch:
    #             if word != word_unk_id:
    #                 n_words += 1
    #     assert n_words < chat_config.batch_size * chat_config.max_len
    #
    #     total_words+=n_words
    #
    #     loss = emotion_chat_machine.valid_step(this_post_data, this_post_len, this_train_res_data,
    #                                        this_predict_res_data, n_words, this_emotion_labels, this_emotion_mask)
    #     entropy_loss, reg_loss, total_loss = loss
    #     total_entropy_loss+=entropy_loss*chat_config.batch_size
    #     # logger.info("epoch=%d, batch=%d, entropy loss=%f, n_words=%d, ppl=%f\n" %
    #     #             (restore_model, j+1, entropy_loss, n_words, math.exp(entropy_loss*chat_config.batch_size / n_words)))
    # logger.info("epoch=%d, valid_entropy loss=%f, total_words=%f, ppl=%f\n" %
    #             (restore_model, total_entropy_loss, total_words,math.exp(total_entropy_loss / total_words)))
    #
    # valid_result_out.write(str(restore_model)+'\t'+str(total_entropy_loss)+'\t'+str(total_words)+'\t'+ str(math.exp(total_entropy_loss / total_words))+'\n')


def main(_):
    for restore_model in [8]:
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        train(FLAGS.config_file, FLAGS.pre_train_word_count_file, FLAGS.emotion_words_dir, FLAGS.post_file,
              FLAGS.response_file, FLAGS.emotion_label_file, FLAGS.embedding_file, FLAGS.train_word_count, sess,
              FLAGS.checkpoint_dir, FLAGS.max_vocab_size, FLAGS.test_post_file, FLAGS.test_response_file,
              FLAGS.test_label_file, FLAGS.valid_result_file, restore_model)


if __name__ == "__main__":
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("train_batch.py")))
    data_dir = os.path.join(model_path, "data")

    parse = argparse.ArgumentParser()
    parse.add_argument("--config_file", type=str, default=os.path.join(model_path, "conf/config.txt"),
                       help="configuration file path")
    parse.add_argument("--pre_train_word_count_file", type=str,
                       default=os.path.join(data_dir, "NLPCC2017_95W/word.count.txt"),
                       help="nlp cc word count file")
    parse.add_argument("--emotion_words_dir", type=str, default=os.path.join(data_dir, "emotion_words_human"),
                       help="emotion words directory")
    parse.add_argument("--post_file", type=str, default=os.path.join(data_dir, "NLPCC2017_95W/train.post.data.txt"),
                       help="post file path")
    parse.add_argument("--response_file", type=str,
                       default=os.path.join(data_dir, "NLPCC2017_95W/train.response.data.txt"),
                       help="response file path")
    parse.add_argument("--emotion_label_file", type=str,
                       default=os.path.join(data_dir, "NLPCC2017_95W/train.emotion.labels.txt"),
                       help="emotion label file path")
    parse.add_argument("--embedding_file", type=str,
                       default=os.path.join(model_path, "../pretrained_word_embeddings/sgns.weibo.bigram-char"),
                       help="word embedding file path")
    parse.add_argument("--train_word_count", type=str, default=os.path.join(data_dir, "NLPCC2017_95W/word.count.txt"),
                       help="training count file path")
    parse.add_argument("--unk", type=str, default="</s>", help="symbol for unk words")
    parse.add_argument("--start_symbol", type=str, default="<ss>", help="symbol for response sentence start")
    parse.add_argument("--end_symbol", type=str, default="<es>", help="symbol for response sentence end")
    parse.add_argument("--checkpoint_dir", type=str, default=os.path.join(model_path, "model_1"),
                       help="saving checkpoint directory")
    parse.add_argument("--test_post_file", type=str,
                       default=os.path.join(data_dir, "NLPCC2017_95W/test.post.data.txt"),
                       help="file path for test post")
    parse.add_argument("--test_response_file", type=str,
                       default=os.path.join(data_dir, "NLPCC2017_95W/test.response.data.txt"),
                       help="file path for test response")
    parse.add_argument("--test_label_file", type=str,
                       default=os.path.join(data_dir, "NLPCC2017_95W/test.emotion.labels.txt"))
    parse.add_argument("--generate_response_file", type=str,
                       default=os.path.join(model_path, "result_1/test.response.txt"),
                       help="file path for test response")
    parse.add_argument("--valid_result_file", type=str,
                       default=os.path.join(model_path, "result_1/valid.result.txt"),
                       help="file path for test response")
    parse.add_argument("--stop_words_file", type=str,
                       default=os.path.join(data_dir, "stop_words/stop-word-zh.txt"),
                       help="stop word file path")
    parse.add_argument("--max_vocab_size", type=int, default=100000, help="maximum vocabulary size")

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
