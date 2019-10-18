# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:59:25 2019

@author: lawle
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mnist = input_data.read_data_sets(".", one_hot=True)

tf.reset_default_graph()
latent_dim = 128
max_epochs = 200
b_size = 128

X = tf.placeholder(tf.float32, [None, 784])
X_image = tf.reshape(X, [-1, 28, 28, 1])

z = tf.placeholder(tf.float32, [None, latent_dim])
is_train = tf.placeholder(tf.bool)

def discriminator(x, is_train=False, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        layer1 = tf.layers.conv2d(x, 32, (3,3), strides=(2,2), padding="same", activation=tf.nn.leaky_relu)
        layer1 = tf.contrib.layers.batch_norm(layer1, center=True, scale=True, is_training=is_train)
        layer2 = tf.layers.conv2d(layer1, 64, (3,3), strides=(2,2), padding="same", activation=tf.nn.leaky_relu)
        layer2 = tf.contrib.layers.batch_norm(layer2, center=True, scale=True, is_training=is_train)
        layer3 = tf.layers.conv2d(layer2, 128, (3,3), strides=(2,2), padding="same", activation=tf.nn.leaky_relu)
        layer3 = tf.contrib.layers.batch_norm(layer3, center=True, scale=True, is_training=is_train)
        layer4 = tf.reshape(layer3, [-1, 4*4*128])
        out = tf.layers.dense(layer4, 1, activation="sigmoid")
    
    return out

def generator(z, is_train=False):
    with tf.variable_scope("generator"):
        layer1 = tf.layers.dense(z, 7*7*128, activation="relu")
        layer1 = tf.contrib.layers.batch_norm(layer1, center=True, scale=True, is_training=is_train)
        layer1 = tf.reshape(layer1, [-1, 7,7,128])
        layer2 = tf.layers.conv2d_transpose(layer1, 64, (3,3), strides=(2,2), padding="same", activation="relu")
        layer2 = tf.contrib.layers.batch_norm(layer2, center=True, scale=True, is_training=is_train)
        layer3 = tf.layers.conv2d_transpose(layer2, 32, (3,3), strides=(2,2), padding="same", activation="relu")
        layer3 = tf.contrib.layers.batch_norm(layer3, center=True, scale=True, is_training=is_train)
        out = tf.layers.conv2d_transpose(layer3, 1, (3,3), strides=(1,1), padding="same", activation="sigmoid")

    return out

generate_image = generator(z, is_train=is_train)
discriminate_origin_image = discriminator(X_image)
discriminate_fake_image = discriminator(generate_image, reuse=True)

d_loss = tf.reduce_mean(1/2 * tf.square(discriminate_origin_image-1) + 1/2 * tf.square(discriminate_fake_image))
g_loss = tf.reduce_mean(1/2* tf.square(discriminate_fake_image-1))

vars = tf.trainable_variables()
d_vars = [var for var in vars if "discriminator" in var.name]
g_vars = [var for var in vars if "generator" in var.name]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    d_optimizer = tf.train.AdamOptimizer(0.001).minimize(d_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(0.001).minimize(g_loss, var_list=g_vars)

num_batch = int(mnist.train.num_examples/b_size)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for e in range(max_epochs):
    d_t_loss = 0
    g_t_loss = 0
    
    for _ in range(num_batch):
        b_x, _ = mnist.train.next_batch(b_size)
        b_z = np.random.uniform(-1, 1, [b_size, latent_dim])
        _, d_b_loss = sess.run([d_optimizer, d_loss], feed_dict={X:b_x, z:b_z, is_train:True})
        _, g_b_loss = sess.run([g_optimizer, g_loss], feed_dict={X:b_x, z:b_z, is_train:True})
        
        d_t_loss += d_b_loss
        g_t_loss += g_b_loss
    
    print("%s epochs, d loss: %.3f, g loss: %.3f" % (e, d_t_loss/num_batch, g_t_loss/num_batch))
    
    t_z = np.random.uniform(-1, 1, [25, latent_dim])
    generated_image = sess.run(generate_image, feed_dict={z:t_z, is_train:False})
    
    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, image in enumerate(generated_image):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(np.reshape(image, (28,28)), cmap="Greys")
        
    plt.show()
    if e % 20 == 0:
        fig.savefig("%d epochs.jpg" % e)
