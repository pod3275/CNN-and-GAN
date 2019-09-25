##### for jupyter notebook & lab ====> %matplotlib inline

# Reference
#   https://gist.github.com/solaris33/908e23c4e4324658b93721caa7a6a097

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import os
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras import backend as K

############################# Originally here - tensorflow.keras.datasets.cifar10
def load_data():
  """Loads CIFAR10 dataset.
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  # dirname = 'cifar-10-batches-py'
  # origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  # path = get_file(dirname, origin=origin, untar=True)
  path = "./cifar-10-batches-py/"                                       ### Modify it

  print(path)
  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)
############################# Originally here - tensorflow.keras.datasets.cifar10

(X_train, Y_train), (X_test, Y_test) = load_data()
Y_train_one_hot = tf.squeeze(tf.one_hot(Y_train, 10), axis=1)
Y_test_one_hot = tf.squeeze(tf.one_hot(Y_test, 10), axis=1)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))
L1 = tf.nn.relu(tf.nn.conv2d(X, filter=W1, strides=[1, 1, 1, 1, ], padding='SAME'))
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
L2 = tf.nn.relu(tf.nn.conv2d(L1, filter=W2, strides=[1, 1, 1, 1, ], padding='SAME'))
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.reshape(L2, [-1, 8 * 8 * 64])
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([8 * 8 * 64, 512], stddev=0.01))
L3 = tf.nn.relu(tf.matmul(L2, W3))
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([512, 10], stddev=0.01))
B4 = tf.Variable(tf.random_normal(shape=[10], stddev=0.01))
model = tf.nn.softmax(tf.matmul(L3, W4) + B4)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(model, 1e-10, 1.0)), [1]))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 128
def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

for epoch in range(15):
    total_cost = 0

    total_batch = int(len(X_train)/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, X_train, Y_train_one_hot.eval(session=sess))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

    batch_xs, batch_ys = next_batch(10000, X_test, Y_test_one_hot.eval(session=sess))
    print('Test Acc. = ', sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}))

print('Training Done!')
batch_xs, batch_ys = next_batch(10000, X_test, Y_test_one_hot.eval(session=sess))
print('Test Acc. = ', sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}))

sess.close()