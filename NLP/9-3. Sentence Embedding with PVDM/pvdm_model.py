# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn

# Write your code here
class Model(object):
    def __init__(self, window=5, emb_dim=100, vocab_size=10000, num_samples=64, use_clip=True, learning_rate=0.01, end_token=0):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.window = window
        self.emb_dim = emb_dim

        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.use_clip = use_clip
        self.learning_rate = learning_rate
        self.end_token = end_token

        # Placeholder
        self.x = tf.placeholder(dtype=tf.int32, shape=(None, self.window))
        self.pid = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.target = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self._target = tf.reshape(self.target, [-1, 1])

        self.emb_W = self.get_var(name='emb_W', shape=[self.vocab_size, self.emb_dim])
        self.input_emb = tf.nn.embedding_lookup(self.emb_W, self.x)

        self.emb_P = self.get_var(name='emb_P', shape=[5010, self.emb_dim])
        self.paragraph_emb = tf.nn.embedding_lookup(self.emb_P, self.pid)

        self.concatnated_layer = tf.concat([self.paragraph_emb, tf.reshape(self.input_emb, [-1, self.window * self.emb_dim])], axis=1)

        self.build_loss()
        self.build_opt()
        self.build_finder()

    def build_loss(self):
        self.nce_w = self.get_var(name="nce_w", shape=[self.vocab_size, self.emb_dim * 6])
        self.nce_b = self.get_var(name="nce_b", shape=[self.vocab_size])

        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.nce_w, biases=self.nce_b,
                                       labels=self._target,
                                       inputs=self.concatnated_layer,
                                       num_sampled=self.num_samples,
                                       num_classes=self.vocab_size))

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

    def build_finder(self):
        self.finder_sample = tf.placeholder(dtype=tf.float32, shape=(1, self.emb_dim))
        self.finder_train = self.emb_P

        a_val = tf.sqrt(tf.reduce_sum(tf.square(self.finder_sample), axis=1))
        b_val = tf.sqrt(tf.reduce_sum(tf.square(self.finder_train), axis=1))
        denom = a_val * b_val
        num = tf.reduce_sum(self.finder_sample * self.finder_train, axis=1)
        cos_similarity = tf.clip_by_value(num / (denom + 1e-16), 0, 1)
        self.similar_values, self.similar_idx = tf.nn.top_k(cos_similarity, k=10)

    def leaky_relu(self, x):
        return tf.maximum((x), 0.1*(x))

    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)


    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/encdec", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/encdec")
        print(' * model restored ')