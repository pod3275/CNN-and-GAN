# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time

from cnn_model import Model
from data_loader import text_data
data = text_data("../dataset/ptb/")


def initialize_session():
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
BATCH_SIZE = 20         # 배치 사이즈
num_k = 7               # 앞에 볼 단어 개수
emb_dim = 64            # 단어 embedding dimension
learning_rate = 0.0005  # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
##################################################

model = Model(num_k=num_k, emb_dim=emb_dim, vocab_size=data.vocab_size,
              use_clip=True, learning_rate=learning_rate)

sess = initialize_session()
sess.run(tf.global_variables_initializer())


def sample_test(test_input=""):
    # test_input = raw_input("test text: ") # input("test text: ") for python 2, 3
    test_x = np.zeros((1, num_k), dtype=np.int32)
    words = test_input.split()
    for i in range(min(num_k, len(words))):
        test_x[0][-i-1] = data.w2idx[words[-i-1]]
    out_x = sess.run(model.out_y, feed_dict={model.x: test_x})
    print(out_x[0], data.idx2w[out_x[0]])

def test_model():
    num_it = int(len(data.test_ids) / BATCH_SIZE)
    test_x = np.zeros((BATCH_SIZE, num_k), dtype=np.int32)
    mask = np.zeros(BATCH_SIZE, dtype=np.int32)
    test_loss, test_cnt = 0, 0

    for _ in range(num_it):
        test_ids, length = data.get_test(BATCH_SIZE)
        max_len = max(length)

        test_x.fill(0)
        mask.fill(0)

        for i in range(num_k - 1, max_len - 2):
            for batch in range(len(test_ids)):
                for j in range(0, num_k):
                    if i < j or i - j >= length[batch]:
                        break
                    test_x[batch][num_k - j - 1] = test_ids[batch][i - j]
                mask[batch] = 1 if length[batch] > i+1 else 0
                if length[batch] > i + 1:
                    input_y[batch] = test_ids[batch][i + 1]

            loss = sess.run(model.loss, feed_dict={model.x: test_x, model.y: input_y, model.mask: mask})
            test_loss += loss
            test_cnt += 1
    print("test loss: {:.3f}".format(test_loss / test_cnt))


input_x = np.zeros((BATCH_SIZE, num_k), dtype=np.int32)
input_y = np.zeros(BATCH_SIZE, dtype=np.int32)
input_mask = np.zeros(BATCH_SIZE, dtype=np.int32)
length = np.zeros(BATCH_SIZE, dtype=np.int32)

avg_loss, it_cnt = 0, 0
it_log, it_test, it_save, it_sample = 50, 250, 1000, 250
start_time = time.time()

for it in range(0, 10000):
    train_ids, length = data.get_train(BATCH_SIZE)
    max_len = max(length)
    input_x.fill(0)

    for i in range(num_k - 1, max_len - 2):
        for batch in range(len(train_ids)):
            for j in range(0, num_k):
                if i < j or i-j >= length[batch]:
                    break
                input_x[batch][num_k-j-1] = train_ids[batch][i-j]
            input_mask[batch] = 1 if length[batch] > i+1 else 0

            if length[batch] > i + 1:
                input_y[batch] = train_ids[batch][i+1]

        loss, _ = sess.run([model.loss, model.update],
                           feed_dict={model.x: input_x, model.y: input_y, model.mask: input_mask})
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
        sample_test(test_input="again the specialists were not able to")
