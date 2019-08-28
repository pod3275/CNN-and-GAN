# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

# Write your code here
class Model(object):
    def __init__(self, max_len=200, emb_dim=128, hidden_dim=128, vocab_size=10000,
                 class_size=2, use_clip=True, learning_rate=0.01, end_token="<eos>"):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.end_token = end_token

        self.vocab_size = vocab_size
        self.class_size = 2
        self.use_clip = use_clip
        self.learning_rate = learning_rate

        self.ind = tf.placeholder(dtype=tf.int32, shape=(None, 2))
        self.val = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.y = tf.placeholder(dtype=tf.int32, shape=(None, ))

        # Embedding
        self.batch_size = tf.shape(self.y)[0]

        self.build_model()
        self.build_loss()
        self.build_opt()

    def build_model(self):
        bow = tf.zeros((self.batch_size, self.vocab_size), dtype=tf.float32)

        # bow vector 생성
        bow += tf.scatter_nd(self.ind, self.val, [self.batch_size, self.vocab_size])

        W1 = self.get_var(name="W1", shape=[self.vocab_size, 128], dtype=tf.float32)
        b1 = self.get_var(name="b1", shape=[128], dtype=tf.float32)
        layer_1 = tf.nn.tanh(tf.matmul(bow, W1) + b1)

        W2 = self.get_var(name="W2", shape=[128, 128], dtype=tf.float32)
        b2 = self.get_var(name="b2", shape=[128], dtype=tf.float32)
        layer_2 = tf.nn.tanh(tf.matmul(layer_1, W2) + b2)

        out_W = self.get_var(name="out_W", shape=[128, self.class_size])
        out_b = self.get_var(name="out_b", shape=[self.class_size])
        self.out = tf.nn.softmax(tf.matmul(layer_2, out_W) + out_b)
        self.out_label = tf.argmax(self.out, 1)

    def build_loss(self):
        self.cross_entropy = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.class_size, 1.0, 0.0)
            * tf.log(tf.clip_by_value(tf.reshape(self.out, [-1, self.class_size]), 1e-20, 1.0)), 1)
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

    def leaky_relu(self, x):
        return tf.maximum((x), 0.1*(x))

    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)

    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/cnn", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/cnn")
        print(' * model restored ')