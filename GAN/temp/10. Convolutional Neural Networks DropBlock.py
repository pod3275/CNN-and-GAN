##### for jupyter notebook & lab ====> %matplotlib inline

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def dropblock(x, keep_prob, block_size):
    _,w,h,c = x.shape.as_list()
    gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / ((w - block_size + 1) * (h - block_size + 1))
    sampling_mask_shape = tf.stack([1, h - block_size + 1, w - block_size + 1, c])
    noise_dist = tf.distributions.Bernoulli(probs=gamma)
    mask = noise_dist.sample(sampling_mask_shape)

    br = (block_size - 1) // 2
    tl = (block_size - 1) - br
    pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
    mask = tf.pad(mask, pad_shape)
    mask = tf.nn.max_pool(mask, [1, block_size, block_size, 1], [1, 1, 1, 1], 'SAME')
    mask = tf.cast(1 - mask, tf.float32)

    return tf.multiply(x, mask) * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask)

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

X_image = tf.reshape(X, [-1,28,28,1])

W1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01))
L1 = tf.nn.relu(tf.nn.conv2d(X_image, filter=W1, strides=[1, 1, 1, 1], padding='VALID'))
L1 = dropblock(L1, keep_prob, 3)

W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
L2 = tf.nn.relu(tf.nn.conv2d(L1, filter=W2, strides=[1, 1, 1, 1], padding='VALID'))
L2 = dropblock(L2, keep_prob, 3)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.relu(tf.nn.conv2d(L2, filter=W3, strides=[1, 1, 1, 1], padding='VALID'))
L3 = dropblock(L3, keep_prob, 3)

W4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
L4 = tf.nn.relu(tf.nn.conv2d(L3, filter=W4, strides=[1, 1, 1, 1], padding='VALID'))
L4 = dropblock(L4, keep_prob, 3)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L4 = tf.reshape(L4, [-1, 4 * 4 * 128])

W5 = tf.Variable(tf.random_normal([4 * 4 * 128, 512], stddev=0.01))
L5 = tf.nn.relu(tf.matmul(L4, W5))

W6 = tf.Variable(tf.random_normal([512, 10], stddev=0.01))
B6 = tf.Variable(tf.random_normal(shape=[10], stddev=0.01))
model = tf.nn.softmax(tf.matmul(L5, W6) + B6)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(model, 1e-10, 1.0)), [1]))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('Training Done!')
print('Test Acc. = ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))

sess.close()