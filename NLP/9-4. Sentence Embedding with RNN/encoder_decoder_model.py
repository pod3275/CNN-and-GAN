# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn

# Write your code here
class Model(object):
    def __init__(self, max_len=40, emb_dim=128, hidden_dim=128, vocab_size=10000, use_clip=True, learning_rate=0.01, end_token=0):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = vocab_size
        self.use_clip = use_clip
        self.learning_rate = learning_rate
        self.end_token = end_token

        # Placeholder
        self.x = tf.placeholder(dtype=tf.int32, shape=(None, max_len))
        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.x_len = self.x_len - 1
        self.batch_max_len = tf.reduce_max(self.x_len)

        # sequence mask for different size
        self.batch_size = tf.shape(self.x)[0]
        self.encoder_input = self.x[:,:]
        self.decoder_output = self.x[:,:]
        self.x_mask = tf.sequence_mask(lengths=self.x_len, maxlen=self.batch_max_len, dtype=tf.float32)

        # Embedding
        self.emb_W = self.get_var(name='emb_W', shape=[self.vocab_size, self.emb_dim])
        self.input_emb = tf.nn.embedding_lookup(self.emb_W, self.encoder_input)
        self.output_emb = tf.nn.embedding_lookup(self.emb_W, self.decoder_output)

        self.build_model()
        self.build_loss()
        self.build_opt()
        self.build_finder()

    def build_model(self):
        # Encoder cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # Dynamic encoding
        self.enc_output, self.enc_states = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=self.input_emb,
                                                        sequence_length=self.x_len, dtype=tf.float32)

        self.represented_vector = self.enc_states[1]

        # Output layer
        self.out_layer = Dense(self.vocab_size, dtype=tf.float32, name='out_layer')

        # Decoder cell
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        helper = tf.contrib.seq2seq.TrainingHelper(self.output_emb, self.x_len, time_major=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, self.enc_states, output_layer=self.out_layer)
        outputs, states, length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=self.max_len)
        self.logits = outputs.rnn_output
        self.output = tf.argmax(self.logits, 2)

    def build_loss(self):
        target_labels = self.decoder_output[:, :self.batch_max_len]
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels, logits=self.logits)
        self.loss = tf.reduce_sum(self.cross_entropy * self.x_mask) / (tf.reduce_sum(self.x_mask) + 1e-10)

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
        self.finder_sample = tf.placeholder(dtype=tf.float32, shape=(1, 128))
        self.finder_train = tf.placeholder(dtype=tf.float32, shape=(None, 128))

        a_val = tf.sqrt(tf.reduce_sum(tf.square(self.finder_sample), axis=1))
        b_val = tf.sqrt(tf.reduce_sum(tf.square(self.finder_train), axis=1))
        denom = a_val * b_val
        num = tf.reduce_sum(self.finder_sample * self.finder_train, axis=1)
        cos_similarity = tf.clip_by_value(num / (denom + 1e-16), 0, 1)
        self.similar_values, self.similar_idx = tf.nn.top_k(cos_similarity, k=3)

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