
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import math
import nltk
import os
import random

import time
import json

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import data_utils
import seq2seq_model
import ConfigParser
import codecs

os.environ["CUDA_VISIBLE_DEVICES"] = "9"  

config = ConfigParser.ConfigParser()
config.read('config')

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


tf.app.flags.DEFINE_string("data_dir", "./data/NLPCC2017_95W", "Data directory")

tf.app.flags.DEFINE_string("train_dir", "train/NLPCC2017_95W_ECM", "Training directory.")

tf.app.flags.DEFINE_string("pretrain_dir", "", "Pretraining directory.")

tf.app.flags.DEFINE_integer("pretrain", -1, "pretrain model number")


tf.app.flags.DEFINE_integer("load_model", 0, "which model to load.")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("beam_size", 20, "Size of beam.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("emotion_size", 100, "Size of emotion embedding.")
tf.app.flags.DEFINE_integer("imemory_size", 256, "Size of imemory.")
tf.app.flags.DEFINE_integer("category", 6, "category of emotions.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("post_vocab_size", 20000, "post vocabulary size.")
tf.app.flags.DEFINE_integer("response_vocab_size", 20000, "response vocabulary size.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 7347,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_emb", False,
                            "use embedding model")
tf.app.flags.DEFINE_boolean("use_imemory", False,
                            "use imemory model")
tf.app.flags.DEFINE_boolean("use_ememory", False,
                            "use ememory model")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("beam_search", True, "beam search")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS



_buckets = [(12, 12), (16, 16), (20, 20), (30, 30)]


def read_data(path, max_size=None):
    data_set = [[] for _ in _buckets]
    data = json.load(open(path, 'r'))
    counter = 0
    size_max = 0
    for pair in data:
        post = pair[0]
        responses = pair[1]
        
        
        
        source_ids = [int(x) for x in post[0]]
        
        response = responses
        if not max_size or counter < max_size:
            counter += 1
            if counter % 100000 == 0:
                print("    reading data pair %d" % counter)
                sys.stdout.flush()
            target_ids = [int(x) for x in response[0]]
            target_ids.append(data_utils.EOS_ID)
            
            
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids, int(post[1]), int(response[1])])
                    break
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    return data_set


def refine_data(data):
    new_data = []
    for d in data:
        b = []
        for e in range(6):
            b.append([x for x in d if x[-1] == e])
        new_data.append(b)
    return new_data


def create_model(session, forward_only, beam_search):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.post_vocab_size,
        FLAGS.response_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        embedding_size=FLAGS.embedding_size,
        forward_only=forward_only,
        beam_search=beam_search,
        beam_size=FLAGS.beam_size,
        category=FLAGS.category,
        use_emb=FLAGS.use_emb,
        use_imemory=FLAGS.use_imemory,
        use_ememory=FLAGS.use_ememory,
        emotion_size=FLAGS.emotion_size,
        imemory_size=FLAGS.imemory_size,
        dtype=dtype)
    see_variable = True
    if see_variable == True:
        for i in tf.all_variables():
            print(i.name, i.get_shape())
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
    if ckpt:  
        if FLAGS.load_model == 0:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            path = ckpt.model_checkpoint_path[:ckpt.model_checkpoint_path.find('-') + 1] + str(FLAGS.load_model)
            print("Reading model parameters from %s" % path)
            model.saver.restore(session, path)
    else:
        if pre_ckpt:
            session.run(tf.initialize_variables(model.initial_var))
            if FLAGS.pretrain > -1:
                path = pre_ckpt.model_checkpoint_path[:pre_ckpt.model_checkpoint_path.find('-') + 1] + str(
                    FLAGS.pretrain)
                print("Reading pretrain model parameters from %s" % path)
                model.pretrain_saver.restore(session, path)
            else:
                print("Reading pretrain model parameters from %s" % pre_ckpt.model_checkpoint_path)
                model.pretrain_saver.restore(session, pre_ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
            vec_post, vec_response = data_utils.get_data(FLAGS.data_dir, FLAGS.post_vocab_size,
                                                         FLAGS.response_vocab_size)
            print(vec_post.shape)
            initvec_post = tf.constant(vec_post, shape=vec_post.shape, dtype=dtype, name='init_wordvector_post')
            initvec_response = tf.constant(vec_response, dtype=dtype, name='init_wordvector_response')
            embedding_post = [x for x in tf.trainable_variables() if
                              x.name == 'embedding_attention_seq2seq/RNN/EmbeddingWrapper/embedding:0'][0]
            embedding_response = [x for x in tf.trainable_variables() if
                                  x.name == 'embedding_attention_seq2seq/embedding_attention_decoder/embedding:0'][0]
            session.run(embedding_post.assign(initvec_post))
            session.run(embedding_response.assign(initvec_response))
        if FLAGS.use_ememory:
            vec_ememory = data_utils.get_ememory(FLAGS.data_dir, FLAGS.response_vocab_size)
            initvec_ememory = tf.constant(vec_ememory, dtype=dtype, name='init_ememory')
            ememory = [x for x in tf.all_variables() if
                       x.name == 'embedding_attention_seq2seq/embedding_attention_decoder/external_memory:0'][0]
            session.run(ememory.assign(initvec_ememory))
    return model


def train():
    print(FLAGS.__flags)
    
    print("Preparing data in %s" % FLAGS.data_dir)
    train_path, dev_path, test_path, _, _ = data_utils.prepare_data(
        FLAGS.data_dir, FLAGS.post_vocab_size, FLAGS.response_vocab_size)

    with tf.Session(config=sess_config) as sess:
        
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, False)

        
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)
        dev_set = read_data(dev_path)
        dev_set = refine_data(dev_set)
        train_set = read_data(train_path, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        print([len(x) for x in dev_set])
        print([len(x) for x in train_set])
        
        
        
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        print(train_buckets_scale)

        
        step_time, loss = 0.0, 0.0
        current_step = 0
        
        
        epoch_steps = 940353 / FLAGS.batch_size
        previous_losses = []
        while True:
            
            
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, decoder_emotions = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, decoder_emotions, bucket_id, False, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            
            if current_step % FLAGS.steps_per_checkpoint == 0:
                
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d (%.2f epoch) learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.global_step.eval() / float(epoch_steps),
                                model.learning_rate.eval(), step_time, perplexity))
                
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0
                
                total_loss = .0
                total_len = .0
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("    eval: empty bucket %d" % (bucket_id))
                        continue
                    bucket_loss = .0
                    bucket_len = .0
                    for e in range(6):
                        len_data = len(dev_set[bucket_id][e])
                        for batch in xrange(0, len_data, FLAGS.batch_size):
                            step = min(FLAGS.batch_size, len_data - batch)
                            model.batch_size = step
                            encoder_inputs, decoder_inputs, target_weights, decoder_emotions = model.get_batch_data(
                                dev_set[bucket_id][e][batch:batch + step], bucket_id)
                            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                         target_weights, decoder_emotions, bucket_id, True, False)
                            bucket_loss += eval_loss * step
                        bucket_len += len_data
                    total_loss += bucket_loss
                    total_len += bucket_len
                    bucket_loss = float(bucket_loss / bucket_len)
                    bucket_ppx = math.exp(bucket_loss) if bucket_loss < 300 else float(
                        "inf")
                    print("    dev_set eval: bucket %d perplexity %.2f" % (bucket_id, bucket_ppx))
                total_loss = float(total_loss / total_len)
                total_ppx = math.exp(total_loss) if total_loss < 300 else float(
                    "inf")
                print("    dev_set eval: bucket avg perplexity %.2f" % (total_ppx))
                sys.stdout.flush()
                model.batch_size = FLAGS.batch_size


def decode():
    try:
        from wordseg_python import Global
    except:
        Global = None

    def split(sent):
        sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
        if Global == None:
            return sent.decode("gbk").split(' ')
        tuples = [(word.decode("gbk"), pos) for word, pos in Global.GetTokenPos(sent)]
        return [each[0] for each in tuples]

    load_models = [7347, 14694, 22041, 29388, 36735, 44082, 51429, 58776, 66123, 73470, 80817,
                   88164, 95511, 102858, 110205, 117552, 124899, 132246, 139593, 146940]
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    
    sentences = codecs.open('./data/ESTC/test_uniq_src.txt', 'r', encoding='utf8').read().split('\n')

    
    
    
    

    

    for i in [1]:
        FLAGS.load_model = load_models[i]
        epoch = epochs[i]
        start_time = time.time()

        
        

        

        writer = codecs.open('./result/NLPCC2017_95W/train_time_ECM/epoch_' + str(epoch) + '_with_no_emo', 'w',
                             encoding='utf8')

        
        
        
        
        
        with tf.Session(config=sess_config) as sess:
            with tf.device("/gpu:9"):
                

                model = create_model(sess, True, FLAGS.beam_search)
                model.batch_size = 1  
                beam_search = FLAGS.beam_search
                beam_size = FLAGS.beam_size
                num_output = 1

                
                post_vocab_path = os.path.join(FLAGS.data_dir,
                                               config.get('data', 'post_vocab_file') % FLAGS.post_vocab_size)
                response_vocab_path = os.path.join(FLAGS.data_dir,
                                                   config.get('data',
                                                              'response_vocab_file') % FLAGS.response_vocab_size)
                post_vocab, _ = data_utils.initialize_vocabulary(post_vocab_path)
                _, rev_response_vocab = data_utils.initialize_vocabulary(response_vocab_path)

                
                
                
                

                random.seed(888)
                count = 0
                total_loss = 0
                n_words = 0
                for s in sentences:
                    if len(s) == 0:
                        continue
                    sentence, src_emo, tgt, tgt_emo = s.strip().split('\t')
                    tgt_emo = int(tgt_emo)
                    
                    
                    sentence = " ".join(split(sentence))
                    
                    token_ids = data_utils.sentence_to_token_ids(sentence, post_vocab)
                    int2emotion = ['null', 'like', 'sad', 'disgust', 'angry', 'happy']

                    
                    for decoder_emotion in [tgt_emo]:
                        
                        bucket_candidates = [b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)]
                        if len(bucket_candidates) == 0:
                            token_ids = token_ids[:_buckets[-1][0] - 1]

                        bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
                        
                        encoder_inputs, decoder_inputs, target_weights, decoder_emotions = model.get_batch_data(
                            [[token_ids, [], 0, decoder_emotion]], bucket_id)
                        
                        results, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                               target_weights, decoder_emotions, bucket_id, True,
                                                               beam_search)

                        if beam_search:
                            result = results[0]
                            symbol = results[1]
                            parent = results[2]
                            
                            
                            
                            
                            res = []
                            nounk = []
                            for i, (prb, _, prt) in enumerate(result):
                                
                                if len(prb) == 0: continue
                                for j in xrange(len(prb)):
                                    p = prt[j]
                                    s = -1
                                    output = []
                                    for step in xrange(i - 1, -1, -1):
                                        s = symbol[step][p]
                                        p = parent[step][p]
                                        output.append(s)
                                    output.reverse()
                                    if data_utils.UNK_ID in output:
                                        res.append([prb[j][0], " ".join(
                                            [tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                                    else:
                                        nounk.append([prb[j][0], " ".join(
                                            [tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                            res.sort(key=lambda x: x[0], reverse=True)
                            nounk.sort(key=lambda x: x[0], reverse=True)
                            if len(nounk) < beam_size:
                                res = nounk + res[:(num_output - len(nounk))]
                            else:
                                res = nounk
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            for i in res[:num_output]:
                                total_loss += i[0]
                                n_words += len(i[1].split())
                                count += 1
                                print('\rfinished ' + str(count) + '/' + str(len(sentences)), end='')
                                
                                
                                
                                
                                writer.write(i[1] + '\n')
                        else:

                            
                            outputs = [
                                int(np.argmax(np.split(logit, [2, FLAGS.response_vocab_size], axis=1)[1], axis=1) + 2)
                                for
                                logit in output_logits]
                            
                            if data_utils.EOS_ID in outputs:
                                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                            
                            print(int2emotion[decoder_emotion] + ': ' + "".join(
                                [tf.compat.as_str(rev_response_vocab[output]) for output in outputs]))
                    sys.stdout.flush()

                end_time = time.time()
                print("total time: ", end_time - start_time)
                print((end_time - start_time) / len(sentences), "s/per sentence")


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
