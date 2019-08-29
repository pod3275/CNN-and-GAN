# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time

from pvdm_model import Model
from data_loader_pvdm import text_data

def initialize_session():
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
max_len = 40            # sequence 단어 수 제한
window = 5              # 예측에 사용 할 단어의 수
num_samples = 1024      # sampling에 사용할 수
emb_dim = 100            # 단어 embedding dimension
learning_rate = 0.005   # Learning rate
##################################################

END_TOKEN = "<eos>"
data = text_data("../dataset/ptb", max_len=max_len, window=window, end_token=END_TOKEN)
model = Model(window=window, emb_dim=emb_dim, vocab_size=data.vocab_size, num_samples=num_samples,
              learning_rate=learning_rate, end_token=data.w2idx[END_TOKEN])

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def cosine_similiarty(x, y):
    return np.sum(x * y) / (np.sum(x**2)**0.5 * np.sum(y**2)**0.5)

def sample_finding_test(test_input=""):
    # test_input = raw_input("test text: ") # input("test text: ") for python 2, 3
    words = test_input.split()
    input_x = np.zeros((1, max_len), dtype=np.int32)
    for i, word in enumerate(words):
        if i == max_len:
            break
        input_x[0][i] = data.w2idx[word]

    sample_ids = []
    sample_target = []
    sample_paragraph_ids = []
    for word_idx in range(len(input_x[0])):
        if word_idx + window + 1 >= max_len or input_x[0][word_idx + window + 1] == 0:
            break
        else:
            sample_ids.append(input_x[0][word_idx:word_idx + window])
            sample_target.append(input_x[0][word_idx + window + 1])
            sample_paragraph_ids.append(5001)

    for i in range(10):
        _, sample_pv = sess.run([model.update, model.paragraph_emb],
                                feed_dict = {model.pid: sample_paragraph_ids, model.x: sample_ids, model.target: sample_target})

    sample_pv = sample_pv[-1]

    similar_values, similar_idx = sess.run([model.similar_values, model.similar_idx],
                                           feed_dict={model.finder_sample: [sample_pv]})

    sim_cnt = 0
    for value, idx in zip(similar_values, similar_idx):
        if idx > 5000:
            print(idx, "is passed.")
            continue
        else:
            print(value, all_train_text[idx][:-1])
            sim_cnt = sim_cnt+1
        if sim_cnt==3:
            break

it_log, it_save, it_sample = 100, 50000, 1000
start_time = time.time()

all_train_text = data.get_train_all_text()

avg_loss = 0
for it in range(0, 40000):
    train_pid, train_ids, train_target = data.get_train_each_paragraph()

    if train_ids.size != 0:
        loss, _ = sess.run([model.loss, model.update],
                           feed_dict={model.pid: train_pid, model.x: train_ids, model.target: train_target})

        avg_loss += loss

    if it % it_log == 0:
        print(" it: {:4d} | loss: {:.3f} - {:.2f}s".format(it, avg_loss / it_log, time.time() - start_time))
        avg_loss = 0

    if it % it_save == 0 and it > 0:
        model.save(sess)
    if it % it_sample == 0 and it > 0:
        print("Input Text: there is no asbestos in our products now ")
        print("Similar Text: ")
        sample_finding_test(" there is no asbestos in our products now ")

sess.close()
