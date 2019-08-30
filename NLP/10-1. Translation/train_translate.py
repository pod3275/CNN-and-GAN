# -*- coding: utf-8 -*- #

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import time

from translation_model import Model
from ko2en_loader import Data

def initialize_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    return tf.Session(config=config)

##################################################
BATCH_SIZE = 8         # 배치 사이즈
emb_dim = 64            # 단어 embedding dimension
hidden_dim = 128        # RNN hidden dim
enc_max_len = 60        # sequence 단어 수 제한
dec_max_len = 60        # sequence 단어 수 제한
enc_max_vocab = 20000
dec_max_vocab = 20000

learning_rate = 0.005   # Learning rate
use_clip = True         # Gradient clipping 쓸지 여부
##################################################

data = Data(path='./ko2en_big', save_name='ko2en_big.data', refresh_data=True,
            max_enc_len=enc_max_len, max_dec_len=dec_max_len,
            max_enc_vocab=enc_max_vocab, max_dec_vocab=dec_max_vocab)

model = Model(emb_dim=emb_dim, hidden_dim=hidden_dim,
              enc_vocab=data.enc_vocab, dec_vocab=data.dec_vocab,
              max_enc_len=enc_max_len, max_dec_len=dec_max_len,
              stt_idx=data.stt_idx, end_idx=data.eos_idx,
              use_clip=True, learning_rate=learning_rate)

sess = initialize_session()
sess.run(tf.global_variables_initializer())

def sample_test():
    idx, enc, dec, enc_len, dec_len = data.get_batch('train', 1)
    output_token = sess.run(model.output_token, feed_dict={model.x: enc, model.x_len: enc_len})
    
    encoder = [data.idx2w_enc[o] for o in enc[0] if o != data.pad_idx]
    out_line = [data.idx2w_dec[o] for o in output_token[0] if o != data.pad_idx]
    true_line = [data.idx2w_dec[o] for o in dec[0] if o != data.pad_idx]
    
    if(encoder.count(data.eos_token)):
        encoder = encoder[:encoder.index(data.eos_token)]
    if(out_line.count(data.eos_token)):
        out_line = out_line[:out_line.index(data.eos_token)]
    if(true_line.count(data.eos_token)):
        true_line = true_line[:true_line.index(data.eos_token)]
   
    encoder = " ".join(encoder)   
    out_line = " ".join(out_line)
    true_line = " ".join(true_line)
    print("Encoder Input ===> {}".format(encoder))
    print("Decoder True ===> {}".format(true_line))
    print("Decoder Pred ===> {}".format(out_line))
    print("="*90)
    print()


def test_model():
    num_it = 10
    test_loss, test_cnt = 0, 0

    for _ in range(num_it):
        idx, enc, dec, enc_len, dec_len = data.get_batch('test', BATCH_SIZE)
        loss = sess.run(model.loss, feed_dict={model.x: enc, model.x_len: enc_len,
                                               model.y: dec, model.y_len: dec_len})
        test_loss += loss
        test_cnt += 1
    print("test loss: {:.3f}".format(test_loss / test_cnt))

avg_loss, it_cnt = 0, 0
it_log, it_test, it_save, it_sample = 10, 10, 1000, 10
start_time = time.time()

for it in range(0, 10000):
    idx, enc, dec, enc_len, dec_len = data.get_batch('train', BATCH_SIZE)
    loss, _ = sess.run([model.loss, model.update],
                       feed_dict={model.x: enc, model.x_len: enc_len,
                                  model.y: dec, model.y_len: dec_len})

    avg_loss += loss
    it_cnt += 1

    if it % it_log == 0:
        print(" it: {:4d} | loss: {:.3f} - {:.2f}s".format(it, avg_loss / it_cnt, time.time() - start_time))
        avg_loss, it_cnt = 0, 0

    if it % it_save == 0 and it > 0:
        model.save(sess)
    if it % it_sample == 0 and it > 0:
        sample_test()

sess.close()