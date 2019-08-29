# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time

from dnn_model import Model
from imdb_loader import text_data

def initialize_session():
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    return tf.Session(config=config)

##################################################
max_len = 200           # sequence 단어 수 제한
max_vocab = 20000       # maximum 단어 개수
BATCH_SIZE = 10         # 배치 사이즈
emb_dim = 64            # 단어 embedding dimension
hidden_dim = 128        # RNN hidden dim
learning_rate = 0.005   # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
##################################################

END_TOKEN = "<eos>"
data = text_data("../dataset/aclImdb/", max_len=max_len, end_token=END_TOKEN)
model = Model(max_len=max_len,
              emb_dim=emb_dim,
              hidden_dim=hidden_dim,
              vocab_size=data.vocab_size,
              class_size=2,
              use_clip=True, learning_rate=learning_rate, end_token=data.w2idx[END_TOKEN])

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def test_model():
    num_it = int(len(data.test_ids) / BATCH_SIZE)
    num_it = 10
    test_loss, test_cnt = 0, 0

    for _ in range(num_it):
        test_ids, length, label = data.get_test(BATCH_SIZE)
        ind, val = [], []
        for i, ids in enumerate(test_ids):
            for j in range(length[i]):
                ind.append([i, ids[j]])
                val.append(1.0)
        loss = sess.run(model.loss, feed_dict={model.ind: ind, model.val: val, model.y: label})

        test_loss += loss
        test_cnt += 1
    print("test loss: {:.3f}".format(test_loss / test_cnt))

# 0: neg, 1: pos
avg_loss, it_cnt, same = 0, 0, .0
it_log, it_test, it_save, it_sample = 10, 100, 1000, 100
start_time = time.time()

for it in range(0, 10000):
    train_ids, length, label = data.get_train(BATCH_SIZE)
    val, ind = [], []
    for i, ids in enumerate(train_ids):
        for j in range(length[i]):
            ind.append([i, ids[j]])
            val.append(1.0)
    loss, _, out = sess.run([model.loss, model.update, model.out_label],
                            feed_dict={model.ind: ind, model.val: val, model.y: label})

    for i, o in enumerate(out):
        if o == i % 2:
            same += 1
    avg_loss += loss
    it_cnt += 1

    if it % it_log == 0:
        print(" it: {:4d} | loss: {:.3f} | acc: {:.3f} - {:.2f}s".format(
            it, avg_loss / it_cnt, same/BATCH_SIZE/it_log, time.time() - start_time))
        avg_loss, it_cnt, same = 0, 0, .0

    if it % it_test == 0 and it > 0:
        test_model()
    if it % it_save == 0 and it > 0:
        model.save(sess)
