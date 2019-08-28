import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Write your code here
class Model(object):
    def __init__(self, num_k=7, emb_dim=128, vocab_size=10000, use_clip=True, learning_rate=0.01):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.emb_dim = emb_dim
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
        x_emb_4d = tf.expand_dims(self.x_emb, -1)

        conv_f1 = self.get_var(name='conv_f1', shape=[3, self.emb_dim, 1, 256])
        conv_b1 = self.get_var(name='conv_b1', shape=[256])
        conv_l1 = self.leaky_relu(tf.nn.conv2d(x_emb_4d, filter=conv_f1, strides=[1, 1, 1, 1], padding='VALID') + conv_b1)
        
        conv_f2 = self.get_var(name='conv_f2', shape=[3, 1, 256, 256])
        conv_b2 = self.get_var(name='conv_b2', shape=[256])
        conv_l2 = self.leaky_relu(tf.nn.conv2d(conv_l1, filter=conv_f2, strides=[1, 1, 1, 1], padding='VALID') + conv_b2)
        
        conv_f3 = self.get_var(name='conv_f3', shape=[3, 1, 256, 256])
        conv_b3 = self.get_var(name='conv_b3', shape=[256])
        conv_l3 = self.leaky_relu(tf.nn.conv2d(conv_l2, filter=conv_f3, strides=[1, 1, 1, 1], padding='VALID') + conv_b3)

        self.text_vec = tf.reshape(conv_l3, [-1, 256])
        dense_1 = Dense(128, dtype=tf.float32, name='dense_1')
        layer_1 = tf.nn.tanh(dense_1(self.text_vec))

        self.out_layer = Dense(self.vocab_size, dtype=tf.float32, name='out_layer')
        self.word_prob = tf.nn.softmax(self.out_layer(layer_1))

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
        save_path = saver.save(sess, save_path="models/cnn", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/cnn")
        print(' * model restored ')