import tensorflow as tf

# Write your code here
class Model(object):
    def __init__(self, num_k=7, emb_dim=128, vocab_size=10000, use_clip=True, learning_rate=0.01):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.num_k = num_k
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
        self.x_emb_2d = tf.reshape(self.x_emb, [-1, self.emb_dim * self.num_k])

        d1 = 128
        W1 = self.get_var(name="W1", shape=[self.emb_dim * self.num_k, d1])
        b1 = self.get_var(name="b1", shape=[d1])
        layer_1 = tf.nn.tanh(tf.matmul(self.x_emb_2d, W1) + b1)

        d2 = 256
        W2 = self.get_var(name="W2", shape=[d1, d2])
        b2 = self.get_var(name="b2", shape=[d2])
        layer_2 = tf.nn.tanh(tf.matmul(layer_1, W2) + b2)

        out_W = self.get_var(name="out_W", shape=[d2, self.vocab_size])
        out_b = self.get_var(name="out_b", shape=[self.vocab_size])
        self.word_prob = tf.nn.softmax(tf.matmul(layer_2, out_W) + out_b)

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
        save_path = saver.save(sess, save_path="models/dnn", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/dnn")
        print(' * model restored ')