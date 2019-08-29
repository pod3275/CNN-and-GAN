# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time

from encoder_decoder_model import Model
from data_loader import text_data

def initialize_session():
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
max_len = 40            # sequence 단어 수 제한
BATCH_SIZE = 20         # 배치 사이즈 - 1이 아니면 입력 데이터 구성이 어려움
emb_dim = 64            # 단어 embedding dimension
hidden_dim = 128        # RNN hidden dim
learning_rate = 0.005   # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
##################################################

END_TOKEN = "<eos>"
data = text_data("../dataset/ptb", max_len=max_len, end_token=END_TOKEN)
model = Model(emb_dim=emb_dim, hidden_dim=hidden_dim, vocab_size=data.vocab_size,
              use_clip=True, learning_rate=learning_rate, end_token=data.w2idx[END_TOKEN])

sess = initialize_session()
sess.run(tf.global_variables_initializer())


def sample_decoding_test(test_input=""):
    # test_input = raw_input("test text: ") # input("test text: ") for python 2, 3
    words = test_input.split()
    input_x = np.zeros((1, max_len), dtype=np.int32)
    for i, word in enumerate(words):
        if i == max_len:
            break
        input_x[0][i] = data.w2idx[word]

    input_x_len = [i+1]
    output = sess.run(model.output, feed_dict={model.x: input_x, model.x_len: input_x_len})
    line = " ".join([data.idx2w[o] for o in output[0]])
    print(line)

def sample_finding_test(test_input="", all_train_text=[], all_train_ids=[], all_train_length=[]):
    # test_input = raw_input("test text: ") # input("test text: ") for python 2, 3
    words = test_input.split()
    input_x = np.zeros((1, max_len), dtype=np.int32)
    for i, word in enumerate(words):
        if i == max_len:
            break
        input_x[0][i] = data.w2idx[word]

    input_x_len = [i+1]

    sample_represented_vector = sess.run(model.represented_vector, feed_dict={model.x: input_x, model.x_len: input_x_len})
    train_represented_vectors = sess.run(model.represented_vector, feed_dict={model.x: all_train_ids, model.x_len: all_train_length})

    similar_values, similar_idx = sess.run([model.similar_values, model.similar_idx], feed_dict={model.finder_sample: sample_represented_vector, model.finder_train: train_represented_vectors})
    for value, idx in zip(similar_values, similar_idx):
        print(value, all_train_text[idx][:-1])

def test_model():
    num_it = int(len(data.test_ids) / BATCH_SIZE)
    test_loss, test_cnt = 0, 0

    for _ in range(num_it):
        test_ids, length = data.get_test(BATCH_SIZE)
        loss = sess.run(model.loss, feed_dict={model.x: test_ids, model.x_len: length})

        test_loss += loss
        test_cnt += 1
    print("test loss: {:.3f}".format(test_loss / test_cnt))

avg_loss, it_cnt = 0, 0
it_log, it_test, it_save, it_sample = 10, 100, 1000, 100
start_time = time.time()

all_train_text, all_train_ids, all_train_length = data.get_train_all()

for it in range(0, 10000):
    train_ids, length = data.get_train(BATCH_SIZE)
    loss, _ = sess.run([model.loss, model.update],
                       feed_dict={model.x: train_ids, model.x_len: length})
    # Can get text embedding
    # text_vector = sess.run(model.enc_states, feed_dict={model.x: train_ids, model.x_len: length})

    avg_loss += loss
    it_cnt += 1

    if it % it_log == 0:
        print(" it: {:4d} | loss: {:.3f} - {:.2f}s".format(it, avg_loss / it_cnt, time.time() - start_time))
        avg_loss, it_cnt = 0, 0

    if it % it_test == 0 and it > 0:
        test_model()
    if it % it_save == 0 and it > 0:
        model.save(sess)
    if it % it_sample == 0 and it > 0:
        print("Input Text: there is no asbestos in our products now ")
        print("Decoded Text: ")
        sample_decoding_test(" there is no asbestos in our products now ")
        print("Similar Text: ")
        sample_finding_test(" there is no asbestos in our products now ", all_train_text, all_train_ids, all_train_length)

sess.close()
