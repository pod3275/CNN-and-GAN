# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:24:13 2019

@author: lawle
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

max_epochs = 150
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# noise space dimension
N_noise = 128

# input image, noise placeholder
IMAGE = tf.placeholder(tf.float32, [None, 784])
NOISE = tf.placeholder(tf.float32, [None, N_noise])
IS_TRAINING = tf.placeholder(tf.bool) # for batch normalization

# for CNN input
IMAGE_2D = tf.reshape(IMAGE, [-1, 28, 28, 1])


# Build discriminator - no max pooling, leaky_relu
def Discriminator(X, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        D_l1 = tf.layers.conv2d(X, 32, [3, 3], strides=(2, 2), padding='SAME')
        D_l1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(D_l1, is_training=IS_TRAINING, updates_collections = None))

        D_l2 = tf.layers.conv2d(D_l1, 64, [3, 3], strides=(2, 2), padding='SAME')
        D_l2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(D_l2, is_training=IS_TRAINING, updates_collections = None))

        D_l3 = tf.layers.conv2d(D_l2, 128, [3, 3], strides=(2, 2), padding='SAME')
        D_l3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(D_l3, is_training=IS_TRAINING, updates_collections = None))
        D_l3_flatten = tf.contrib.layers.flatten(D_l3)

        D_out = tf.layers.dense(D_l3_flatten, 1, activation=tf.nn.sigmoid)

        return D_out


# Build generator - no max pooling, conv2d_traspose
def Generator(X):
    with tf.variable_scope('Generator'):
        G_l1 = tf.layers.dense(X, 128 * 7 * 7)
        G_l1_2D = tf.reshape(G_l1, [-1, 7, 7, 128])
        G_l1_2D = tf.nn.relu(tf.contrib.layers.batch_norm(G_l1_2D, is_training=IS_TRAINING, updates_collections = None))

        G_l2 = tf.layers.conv2d_transpose(G_l1_2D, 64, [3, 3], strides=(2, 2), padding='SAME')
        G_l2 = tf.nn.relu(tf.contrib.layers.batch_norm(G_l2, is_training=IS_TRAINING, updates_collections = None))

        G_l3 = tf.layers.conv2d_transpose(G_l2, 32, [3, 3], strides=(2, 2), padding='SAME')
        G_l3 = tf.nn.relu(tf.contrib.layers.batch_norm(G_l3, is_training=IS_TRAINING, updates_collections = None))

        G_l4 = tf.layers.conv2d_transpose(G_l3, 1, [3, 3], strides=(1, 1), padding='SAME')
        G_l4 = tf.nn.sigmoid(G_l4)

        G_out = G_l4

        return G_out


Generated_image = Generator(NOISE)
Discriminate_real_image = Discriminator(IMAGE_2D)
Discriminate_generated_image = Discriminator(Generated_image, reuse=True)


# training loss of GAN - G and D each
D_loss = -tf.reduce_mean(tf.log(Discriminate_real_image) + tf.log(1 - Discriminate_generated_image))
# or, for label smoothing,
#D_loss = -tf.reduce_mean(0.9 * tf.log(Discriminate_real_image) + tf.log(1 - Discriminate_generated_image))
G_loss = -tf.reduce_mean(tf.log(Discriminate_generated_image))


# optimizers - G and D each
vars = tf.trainable_variables()
D_vars = [var for var in vars if 'Discriminator' in var.name]
G_vars = [var for var in vars if 'Generator' in var.name]

D_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_loss, var_list=D_vars)
G_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=G_vars)


# session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 128
total_batch = int(mnist.train.num_examples / batch_size)


# training phase
for epoch in range(max_epochs):
    d_total_loss = 0
    g_total_loss = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_noises = np.random.uniform(-1., 1., [batch_size, N_noise])

        # update D-G
        _, d_loss_val = sess.run([D_optimizer, D_loss], feed_dict={IMAGE: batch_xs, NOISE: batch_noises, IS_TRAINING: True})
        _, g_loss_val = sess.run([G_optimizer, G_loss], feed_dict={NOISE: batch_noises, IS_TRAINING: True})

        d_total_loss += d_loss_val
        g_total_loss += g_loss_val

    print('Epoch:', '%04d' % epoch, 'D loss =', '{:.3f}'.format(d_total_loss / total_batch), 'G loss =', '{:.3f}'.format(g_total_loss / total_batch))

    # Performance check : display generated images
    if epoch % 10 == 0:
        
        # randomly generate 25 images
        g_samples = sess.run(Generated_image, feed_dict={NOISE: np.random.uniform(-1., 1., [25, N_noise]), IS_TRAINING: False})

        # display images
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(g_samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            plt.imshow(np.reshape(sample, (28, 28)), cmap='Greys')

        plt.show()


        # linear interpolation of two different generated images
        linear_sample_start = np.random.uniform(-1., 1., [N_noise])
        linear_sample_end = np.random.uniform(-1., 1., [N_noise])
        linear_sample_step = (linear_sample_end - linear_sample_start) / 24
        linear_samples = []

        for i in range(25):
            linear_samples.append(linear_sample_start + linear_sample_step * i)

        g_samples = sess.run(Generated_image, feed_dict={NOISE: linear_samples, IS_TRAINING: False})

        # display images
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(g_samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            plt.imshow(np.reshape(sample, (28, 28)), cmap='Greys')

        plt.show()

sess.close()