# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:42:19 2019

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


# Build discriminator
with tf.variable_scope('Discriminator'):
    # Discriminator: 2 layers FCNN (image->2class)
    D_w1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
    D_b1 = tf.Variable(tf.random_normal([256], stddev=0.01))

    D_w2 = tf.Variable(tf.random_normal([256, 1], stddev=0.01))
    D_b2 = tf.Variable(tf.random_normal([1], stddev=0.01))

def Discriminator(X):
    D_l1 = tf.nn.relu(tf.matmul(X, D_w1) + D_b1)
    D_out = tf.nn.sigmoid(tf.matmul(D_l1, D_w2) + D_b2)
    return D_out

discriminate = Discriminator(IMAGE) # Real: 1, fake: 0


# Build generator
with tf.variable_scope('Generator'):
    # Generator: 2 layers FCNN (noise->image)
    G_w1 = tf.Variable(tf.random_normal([N_noise, 256], stddev=0.01))
    G_b1 = tf.Variable(tf.random_normal([256], stddev=0.01))

    G_w2 = tf.Variable(tf.random_normal([256, 784], stddev=0.01))
    G_b2 = tf.Variable(tf.random_normal([784], stddev=0.01))

def Generator(X):
    G_l1 = tf.nn.relu(tf.matmul(X, G_w1) + G_b1)
    G_out = tf.nn.sigmoid(tf.matmul(G_l1, G_w2) + G_b2)
    return G_out

Generated_image = Generator(NOISE)


# training loss of GAN - G and D each
D_loss = -tf.reduce_mean(tf.log(Discriminator(IMAGE)) + tf.log(1 - Discriminator(Generator(NOISE))))
# G_loss = tf.reduce_mean(tf.log(1 - Discriminator(Generator(NOISE))))

# or, below is better (avoid gradient flat when Generator samples bad)
G_loss = -tf.reduce_mean(tf.log(Discriminator(Generator(NOISE))))


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


# Training phase
for epoch in range(max_epochs):
    d_total_loss = 0
    g_total_loss = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_noises = np.random.uniform(-1., 1., [batch_size, N_noise])

        # update D-G
        _, d_loss_val = sess.run([D_optimizer, D_loss], feed_dict={IMAGE: batch_xs, NOISE: batch_noises})
        _, g_loss_val = sess.run([G_optimizer, G_loss], feed_dict={NOISE: batch_noises})

        d_total_loss += d_loss_val
        g_total_loss += g_loss_val

    print('Epoch:', '%04d' % epoch, 'D loss =', '{:.3f}'.format(d_total_loss / total_batch), 'G loss =', '{:.3f}'.format(g_total_loss / total_batch))

    # Performance check : display generated images
    if epoch % 10 == 0:
        
        # randomly generate 25 images
        g_samples = sess.run(Generated_image, feed_dict={NOISE: np.random.uniform(-1., 1., [25, N_noise])})

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

        g_samples = sess.run(Generated_image, feed_dict={NOISE: linear_samples})

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