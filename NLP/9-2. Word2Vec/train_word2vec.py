# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time, random

from word2vec_model import W2V
from imdb_loader import text_data

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

##################################################
BATCH_SIZE = 128        # 배치 사이즈
max_len = 200           # text 최대 길이
window_size = 4         # 앞 뒤 window size
max_vocab = 10000       # maximum 단어 개수
emb_dim = 128           # 단어 embedding dimension
num_samples = 1024      # samples numbers for sampled softmax or NCE
learning_rate = 0.001   # Learning rate
use_clip = False        # Gradient clipping 쓸지 여부
##################################################

END_TOKEN = "<eos>"
data = text_data("./dataset", max_len=max_len, max_vocab=max_vocab, end_token=END_TOKEN)
with tf.variable_scope("w2v"):
    w2v = W2V(emb_dim=emb_dim, vocab_size=data.vocab_size, num_samples=num_samples,
              use_clip=True, learning_rate=learning_rate)

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def cosine_similiarty(x, y):
    return np.sum(x * y) / (np.sum(x**2)**0.5 * np.sum(y**2)**0.5)

def similar_words(word="good"):
    idx = data.w2idx[word]
    emb_w = np.array(sess.run(w2v.emb_w))

    scores = []
    for i in range(len(emb_w)):
        cosine_sim = cosine_similiarty(emb_w[i], emb_w[idx])
        scores.append((cosine_sim, i))
    scores = sorted(scores, reverse=True)

    print("\nSimilar words with [{}]".format(word))
    for i in range(1, 10):
        cosine_sim, id = scores[i]
        print("{:2d} - {}: {:.4f}".format(i, data.idx2w[id], cosine_sim))


start_time = time.time()
inputs_labels = []
for _, ids in enumerate(data.train_ids):
    length = data.train_len[_] - 1 # except <eos>
    for i in range(0, length):
        # i 이전의 window size
        for j in range(max(0, i - window_size), i):
            inputs_labels.append((ids[i], ids[j]))

        # i 이후의 window size
        for j in range(i + 1, min(length, i + window_size + 1)):
            inputs_labels.append((ids[i], ids[j]))

    if _ % 500 == 0:
        print(" Train id: {} | samples: {} | {:.2f}s".format(_, len(inputs_labels), time.time() - start_time))
    if _ == 25000:
        break

total_size = len(inputs_labels)
random.shuffle(inputs_labels)
print(data.vocab_size)
print("Total training samples: {}".format(total_size))

start_time = time.time()
data_point = 0
avg_loss, it_log, it_save, it_sample = .0, 100, 5000, 1000

for it in range(0, 100000):
    _inputs_labels = inputs_labels[data_point: data_point + BATCH_SIZE]
    data_point = (data_point + BATCH_SIZE) % total_size

    _inputs = [_i for _i, _l in _inputs_labels]
    _labels = [_l for _i, _l in _inputs_labels]

    loss, update = sess.run([w2v.loss, w2v.update],
                            feed_dict={w2v.inputs: _inputs, w2v.labels: _labels})
    avg_loss += loss

    if it % it_log == 0 and it:
        print(" it: {:4d} | loss: {:.3f} - {:.2f}s".format(
            it, avg_loss / it_log, time.time() - start_time))
        avg_loss = .0

    if it % it_sample == 0:
        similar_words(word="man")
    if it % it_save == 0 and it > 0:
        w2v.save(sess)
