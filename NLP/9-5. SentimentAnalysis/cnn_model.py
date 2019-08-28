# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

# Write your code here
class Model(object):
    def __init__(self, max_len=200, emb_dim=128, filter_sizes=[2], filter_nums=[100], vocab_size=10000,
                 class_size=2, use_clip=True, learning_rate=0.01, end_token="<eos>"):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.filter_sizes = filter_sizes
        self.filter_nums = filter_nums
        self.end_token = end_token

        self.vocab_size = vocab_size
        self.class_size = 2
        self.use_clip = use_clip
        self.learning_rate = learning_rate

        self.x = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len))
        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.y = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.keep_prob = tf.placeholder_with_default(1.0, shape=None)

        # Embedding
        self.emb_W = self.get_var(name='emb_W', shape=[self.vocab_size, self.emb_dim])
        self.batch_size = tf.shape(self.x)[0]
        self.x_emb = tf.nn.embedding_lookup(self.emb_W,
                                            tf.concat([self.x[:, :-1], tf.ones([self.batch_size, 1], dtype=tf.int32)], 1))

        self.build_model()
        self.build_loss()
        self.build_opt()

    def build_model(self):
        # Dimension change for conv net
        x_emb_4d = tf.expand_dims(self.x_emb, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, filter_num in zip(self.filter_sizes, self.filter_nums):
            # Seperated name scope for variables (W, b) for each filters
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.emb_dim, 1, filter_num]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")

                # Apply nonlinearity
                conv = tf.nn.conv2d(x_emb_4d, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = self.leaky_relu(conv + b)

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, self.max_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(self.filter_nums)
        h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.keep_prob)

        out_W = self.get_var(name="out_W", shape=[num_filters_total, self.class_size])
        out_b = self.get_var(name="out_b", shape=[self.class_size])
        self.out = tf.nn.softmax(tf.matmul(self.h_drop, out_W) + out_b)
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