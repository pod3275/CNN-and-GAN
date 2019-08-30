# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn

# Write your code here
class Model(object):
    def __init__(self, emb_dim=128, hidden_dim=128, attn_dim=256,
                 max_enc_len=50, max_dec_len=50,
                 enc_vocab=5000, dec_vocab=5000,
                 stt_idx=1, end_idx=2,
                 use_clip=True, learning_rate=0.001):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

        self.stt_idx = stt_idx
        self.end_idx = end_idx

        self.use_clip = use_clip
        self.learning_rate = learning_rate

        # Placeholder
        self.x = tf.placeholder(dtype=tf.int32, shape=(None, max_enc_len))
        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None, ))

        self.y = tf.placeholder(dtype=tf.int32, shape=(None, max_dec_len))
        self.y_len = tf.placeholder(dtype=tf.int32, shape=(None, ))

        # sequence mask for different size
        self.batch_size = tf.shape(self.x)[0]
        self.masks = tf.sequence_mask(lengths=self.y_len, maxlen=self.max_dec_len, dtype=tf.float32)

        # Embedding
        self.emb_W_enc = tf.get_variable(name='emb_W_enc', shape=[self.enc_vocab, self.emb_dim], dtype=tf.float32, initializer=self.initializer)
        self.emb_W_dec = tf.get_variable(name='emb_W_dec', shape=[self.dec_vocab, self.emb_dim], dtype=tf.float32, initializer=self.initializer)

        self.x_emb = tf.nn.embedding_lookup(self.emb_W_enc, self.x)
        self.y_emb = tf.nn.embedding_lookup(self.emb_W_dec, self.y)

        self.rnn_encode()
        self.init_enc_attention()

        self.decoder_cell = LSTMCell(self.hidden_dim, state_is_tuple=True)
        self.out_layer = Dense(self.dec_vocab, name='out_layer')
        self.decoder_train()
        self.decoder_infer()

        self.build_loss()
        self.build_opt()

    def rnn_encode(self):
        # Bi-direction rnn encoder (forward, backward)
        (self.enc_out, (fw_state, bw_state)) = bidirectional_dynamic_rnn(
            LSTMCell(self.hidden_dim), LSTMCell(self.hidden_dim),
            inputs=self.x_emb, sequence_length=self.x_len, dtype=tf.float32)
        self.enc_out = tf.concat(self.enc_out, 2)

        # Init for decoder's first state
        decoder_init_c = Dense(self.hidden_dim, name="decoder_c", activation=tf.nn.tanh, bias_initializer=self.initializer)
        decoder_init_h = Dense(self.hidden_dim, name="decoder_h", activation=tf.nn.tanh, bias_initializer=self.initializer)
        self.init_state = LSTMStateTuple(decoder_init_c(tf.concat([fw_state.c, bw_state.c], 1)),
                                         decoder_init_h(tf.concat([fw_state.h, bw_state.h], 1)))

    # 논문: https://arxiv.org/pdf/1409.0473.pdf
    # Attention process: softmax(e_t)
    # e_t,i = v^T * tanh (U*h_i + W*s_t + b)  --> h_i, s_t 는 각각 encoder / decoder state
    # U*h_i + b 는 encoding 만 완료되면 구할 수 있으므로, 미리 구해주고
    # decoder state 는 매 디코딩 과정에서 나오는 state를 사용
    # encoder 각각의 hidden state에 matrix를 개별적을 곱하기 위해 convolution 연산 사용
    def init_enc_attention(self):
        # self.attn_dim = 300   # defined at init
        att_e = tf.layers.dense(self.enc_out, self.attn_dim, kernel_initializer=self.initializer, bias_initializer=self.initializer)
        self.attUeh = tf.reshape(att_e, [-1, self.attn_dim])
        self.attve = tf.get_variable(name='attnV_et', shape=[self.attn_dim, 1], initializer=self.initializer)


    # Encoder attention softmax
    # decoder state 를 넘겨받아서 attention distribution softmax(e_t) 리턴
    # tile: encoder 각 state의 e_t,i 구할 때 decoder state 가 필요하므로 encoder 길이(step)만큼 tiling 해줌
    # tiling 한 뒤 마찬가지로 conv 연산 사용 (W 연산 후 tiling해도 가능할 수 있으나, 직관적으로 위와 똑같이 사용)
    def encoder_attention(self, state):
        # decoder state
        st = tf.tile(tf.expand_dims(state, 1), [1, self.max_enc_len, 1]) ## state : batch_size, dim
        attWes = tf.reshape(tf.layers.dense(st, self.attn_dim, kernel_initializer=self.initializer,
                                            bias_initializer=self.initializer), [-1, self.attn_dim])
        # encoder attention distribution
        e_t = tf.reshape(tf.matmul(tf.nn.tanh(attWes + self.attUeh), self.attve),
                         [self.batch_size, self.max_enc_len])

        return tf.nn.softmax(e_t)


    def decoder_train(self):
        # time-batch-dimension
        y_emb_tbd = tf.transpose(self.y_emb, [1, 0, 2])
        word_prob = tf.TensorArray(dtype=tf.float32, size=self.max_dec_len)

        def body(step, state, word_prob):
            enc_softmax = self.encoder_attention(state.h)
            context_vector = tf.reduce_sum(self.enc_out * tf.expand_dims(enc_softmax, -1), axis=1)
            word_logit = self.out_layer(tf.concat([state.h, context_vector], 1))
            word_prob = word_prob.write(step, word_logit)

            token_emb = y_emb_tbd[step]
            inp = tf.concat([token_emb, context_vector], 1)
            next_out, next_state = self.decoder_cell(inp, state)

            return step + 1, next_state, word_prob

        _step, _state, _word_prob = tf.while_loop(
            cond=lambda t, _state, _word_prob: t < self.max_dec_len,
            body=body,
            loop_vars=(0, self.init_state, word_prob))

        self.train_prob = tf.transpose(_word_prob.stack(), perm=[1, 0, 2])

    def decoder_infer(self):
        word_token = tf.TensorArray(dtype=tf.int32, size=self.max_dec_len)

        def body(step, state, word_token):
            enc_softmax = self.encoder_attention(state.h)
            context_vector = tf.reduce_sum(self.enc_out * tf.expand_dims(enc_softmax, -1), axis=1)

            word_logit = self.out_layer(tf.concat([state.h, context_vector], 1))
            next_token = tf.cast(tf.reshape(tf.argmax(word_logit, 1), [self.batch_size]), tf.int32)
            word_token = word_token.write(step, next_token)

            token_emb = tf.nn.embedding_lookup(self.emb_W_dec, next_token)
            inp = tf.concat([token_emb, context_vector], 1)
            next_out, next_state = self.decoder_cell(inp, state)

            return step + 1, next_state, word_token

        _step, _state, _word_token = tf.while_loop(
            cond=lambda t, _state, _word_token: t < self.max_dec_len,
            body=body,
            loop_vars=(0, self.init_state, word_token))

        self.output_token = tf.transpose(_word_token.stack(), perm=[1, 0])

    def build_loss(self):
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.train_prob)
        self.loss = tf.reduce_sum(self.cross_entropy * self.masks) / (tf.reduce_sum(self.masks) + 1e-10)

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


    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/model", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/model")
        print(' * model restored ')