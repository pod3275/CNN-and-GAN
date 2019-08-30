# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn

# Write your code here
class Model(object):
    def __init__(self, emb_dim=128, hidden_dim=128, vocab_size=100,
                 use_clip=True, learning_rate=0.001):
        self.initializer = tf.random_uniform_initializer(-0.05, 0.05)

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.use_clip = use_clip
        self.learning_rate = learning_rate

        self.T = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.t_len = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.Q = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.q_len = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.A = tf.placeholder(dtype=tf.int32, shape=(None,))

        # Batch size
        self.batch_size = tf.shape(self.T)[0]

        # Embedding
        self.emb_W = tf.get_variable(name='emb_W', shape=[self.vocab_size, self.emb_dim],
                                     dtype=tf.float32, initializer=self.initializer)
        self.t_emb = tf.nn.embedding_lookup(self.emb_W, self.T)
        self.q_emb = tf.nn.embedding_lookup(self.emb_W, self.Q)

        self.build_model()
        self.build_loss()
        self.build_opt()

    def build_model(self):
        # 아래처럼 bidirectional_dynamic_rnn 내에서 LSTM Cell을 호출하면 호출하면서 LSTM 변수가 생성되는데,
        # 동일한 bidirectional_dynamic_rnn 모듈들 내에서 이름이 겹치기 때문에 LSTM의 name을 지정해주거나 variable scope 를 구분해 주어야 함
        # Bi-direction LSTM for Text
        with tf.variable_scope("text_lstm"):
            (self.t_out, (t_fw_state, t_bw_state)) = bidirectional_dynamic_rnn(
                LSTMCell(self.hidden_dim), LSTMCell(self.hidden_dim),
                inputs=self.t_emb, sequence_length=self.t_len, dtype=tf.float32)

        # Bi-direction LSTM for Question
        with tf.variable_scope("question_lstm"):
            (self.q_out, (q_fw_state, q_bw_state)) = bidirectional_dynamic_rnn(
                LSTMCell(self.hidden_dim), LSTMCell(self.hidden_dim),
                inputs=self.t_emb, sequence_length=self.t_len, dtype=tf.float32)

        self.text_question = tf.concat([t_fw_state.h, t_bw_state.h, q_fw_state.h, q_bw_state.h], 1)
        self.out_layer = Dense(self.vocab_size, name='out_layer')
        self.word_prob = self.out_layer(self.text_question)

        self.output = tf.cast(tf.argmax(self.word_prob, 1), tf.int32)

    def build_loss(self):
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.A, logits=self.word_prob)
        self.loss = tf.reduce_mean(self.cross_entropy)

    def build_opt(self):
        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grad, var = zip(*optimizer.compute_gradients(self.loss))

        # gradient clipping
        def clipped_grad(grad):
            return [None if g is None else tf.clip_by_norm(g, 2.5) for g in grad]

        if self.use_clip:
            grad = clipped_grad(grad)

        self.update = optimizer.apply_gradients(zip(grad, var))

    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/model", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

        # Load whole weights

    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/model")
        print(' * model restored ')