# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

# Write your code here
class Model(object):
    def __init__(self, num_k=7, emb_dim=128, hidden_dim=128, vocab_size=10000, use_clip=True, learning_rate=0.01):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.num_k = num_k
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = vocab_size
        self.use_clip = use_clip
        self.learning_rate = learning_rate

        self.x = tf.placeholder(dtype=tf.int32, shape=(None, num_k))
        self.y = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.mask = tf.placeholder(dtype=tf.float32, shape=(None, ))

        # Embedding
        self.emb_W = self.get_var(name='emb_W', shape=[self.vocab_size, self.emb_dim])
        self.x_emb = tf.nn.embedding_lookup(self.emb_W, self.x)

        self.build_model()
        self.build_loss()
        self.build_opt()

    def build_model(self):
        lstm_cell = rnn.BasicLSTMCell(self.hidden_dim)
        output, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.x_emb, dtype=tf.float32)

        # output 의 맨 마지막 hidden state만 사용
        last_output = output[:, self.num_k-1, :]  ## batch_size, time_step, hidden_dim

        out_W = self.get_var(name="out_W", shape=[self.hidden_dim, self.vocab_size])
        out_b = self.get_var(name="out_b", shape=[self.vocab_size])
        self.word_prob = tf.nn.softmax(tf.matmul(last_output, out_W) + out_b)

        self.out_y = tf.argmax(self.word_prob, 1)

    def build_loss(self):
        self.cross_entropy = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.vocab_size, 1.0, 0.0)
            * tf.log(tf.clip_by_value(tf.reshape(self.word_prob, [-1, self.vocab_size]), 1e-20, 1.0)), 1)
        self.loss = tf.reduce_sum(self.cross_entropy * self.mask) / (tf.reduce_sum(self.mask) + 1e-10)

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

    def leaky_relu(self, x):
        return tf.maximum((x), 0.1*(x))

    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)


    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/rnn", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/rnn")
        print(' * model restored ')