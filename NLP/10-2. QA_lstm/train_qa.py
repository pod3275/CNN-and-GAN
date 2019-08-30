# -*- coding: utf-8 -*- #

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time
import random

from babi_loader import QAdata
from qa_lstm_model import Model


def initialize_session():
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


##################################################
BATCH_SIZE = 20         # 배치 사이즈
emb_dim = 64            # 단어 embedding dimension
hidden_dim = 64         # RNN hidden dim

max_t_len = 80
max_q_len = 20

learning_rate = 0.0005    # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
##################################################

data = QAdata(path="./dataset/babi", max_t_len=max_t_len, max_q_len=max_q_len)
model = Model(emb_dim=emb_dim, hidden_dim=hidden_dim, vocab_size=data.vocab_size)

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def sample_test():
    T, Q, A, t_len, q_len = data.get_test(1)
    feed_dict = {model.T: T, model.Q: Q, model.t_len: t_len, model.q_len: q_len}
    output = sess.run(model.output, feed_dict=feed_dict)

    t_line = data.ids2sent(T[0])
    q_line = data.ids2sent(Q[0])
    true_A = data.idx2w[A[0]]
    out_A = data.idx2w[output[0]]

    print(" - Given text: {}".format(t_line))
    print("  - Question: {}".format(q_line))
    print("   - Answer: {} / {} (ground-truth)\n".format(out_A, true_A))


def test_model():
    num_it = 10
    test_loss, test_cnt, test_right = 0, 0, .0

    for _ in range(num_it):
        T, Q, A, t_len, q_len = data.get_test(BATCH_SIZE)
        feed_dict = {model.T: T, model.Q: Q, model.A: A,
                     model.t_len: t_len, model.q_len: q_len}
        loss, output = sess.run([model.loss, model.output], feed_dict=feed_dict)
        test_loss += loss
        test_cnt += 1
        for i, o in enumerate(output):
            if o == A[i]:
                test_right += 1
    print(" * test loss: {:.3f} | acc: {:.3f}\n".format(test_loss / test_cnt, test_right / test_cnt / BATCH_SIZE))



print(" * Data size: ", data.total_size)
print(" * Vocab size: ", data.vocab_size)

avg_loss, it_cnt = 0, 0
it_log, it_test, it_save, it_sample = 10, 50, 1000, 50
start_time = time.time()

for it in range(0, 10000):
    T, Q, A, t_len, q_len = data.get_train(BATCH_SIZE)
    feed_dict = {model.T: T, model.Q: Q, model.A: A,
                 model.t_len: t_len, model.q_len: q_len}
    loss, _ = sess.run([model.loss, model.update], feed_dict=feed_dict)

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
        sample_test()
