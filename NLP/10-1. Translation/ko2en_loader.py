#-*- coding: utf-8 -*-
import os, time
from nltk.tokenize import TweetTokenizer
import numpy as np
import random



class Data(object):

    def __init__(self, path, save_name, refresh_data=False,
                 max_enc_len=50, max_dec_len=50,
                 max_enc_vocab=50, max_dec_vocab=50):
        
        self.tokenizer = TweetTokenizer()
        
        self.start_time = time.time()
        self.path = path + ('/' if path[-1] != '/' else '')
        self.name = save_name
        self.save_name = save_name + '.npy'

        self.max_enc_vocab, self.max_dec_vocab = max_enc_vocab, max_dec_vocab
        self.enc_len, self.enc_vocab = 0, 0
        self.dec_len, self.dec_vocab = 0, 0

        self.text_enc, self.text_dec = [], []
        self.train_size, self.val_size, self.test_size = 0, 0, 0
        
        self.pad_token, self.pad_idx = "<pad>", 0
        self.eos_token, self.eos_idx = "</s>", 1
        self.stt_token, self.stt_idx = "<s>", 2
        self.unk_token, self.unk_idx = "<unk>", 3
        self.idx2w_enc, self.idx2w_dec = {}, {}

        print(' *---- Data Loading ----*')
        print(' | path: {}'.format(path))

        if os.path.exists(self.save_name) and not refresh_data:
            self.load()
            self.train_size = len(self.train_idx)
            self.val_size = len(self.val_idx)
            self.test_size = len(self.test_idx)
        else:
            self.w2idx_enc = self.read_vocab("vocab.ko", "enc")
            self.w2idx_dec = self.read_vocab("vocab.en", "dec")

            # Train
            enc, dec, size = self.read_file("train")
            self.text_enc.extend(enc)
            self.text_dec.extend(dec)
            self.train_size += size

            # Val
            enc, dec, size = self.read_file("test")
            self.text_enc.extend(enc)
            self.text_dec.extend(dec)
            self.val_size += size

            # Test
            enc, dec, size = self.read_file("test")
            self.text_enc.extend(enc)
            self.text_dec.extend(dec)
            self.test_size += size

            self.enc_len , self.dec_len = max_enc_len, max_dec_len
            self.enc_ids, self.dec_ids, self.length = self.build_data()
            self.save()

        self.enc_vocab, self.dec_vocab = len(self.w2idx_enc), len(self.w2idx_dec)
        self.size = self.train_size + self.val_size + self.test_size
        for word in self.w2idx_enc:
            self.idx2w_enc[self.w2idx_enc[word]] = word
        for word in self.w2idx_dec:
            self.idx2w_dec[self.w2idx_dec[word]] = word

        print(' | Size: {}, {}, {}'.format(self.train_size, self.val_size, self.test_size))
        print(' | Vocab: {} {}'.format(len(self.w2idx_enc), len(self.w2idx_dec)))
        print(' | Length: {} {}'.format(self.enc_len, self.dec_len))
        print(' | Avg length: {:.1f} / {:.1f} words'.format(self.length.mean(0)[0], self.length.mean(0)[1]))
        print(' | Building time: {:.2f}s'.format(time.time() - self.start_time))
        print(' *---- Dataset Intialized ----\n')

    def ids2sent(self, ids):
        sent = ''
        for i in ids:
            word = self.idx2w[i]
            sent += word + ' '
            if word == self.pad_token:
                break
        return sent.rstrip()

    def word2idx(self, w2idx, word):
        if word in w2idx:
            return w2idx[word]
        else:
            return self.unk_idx

    def read_vocab(self, name="vocab.en", mode="enc"):
        with open(self.path + "/" + name, encoding="utf-8") as fin:
            lines = fin.readlines()
        w2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx, self.stt_token: self.stt_idx, self.eos_token: self.eos_idx}
        for i, line in enumerate(lines):
            word = line.rstrip()
            if word not in w2idx:
                w2idx[word] = len(w2idx)
        return w2idx

    def read_file(self, name="test"):
        with open(self.path + "/" + name + ".ko", encoding="utf-8") as fin:
            enc = fin.readlines()
        with open(self.path + "/" + name + ".en", encoding="utf-8") as fin:
            dec = fin.readlines()

        for line in enc:
            words = line.rstrip().split(" ")
            enc_len = len(words)
            self.enc_len = min(enc_len, self.enc_len)

        for line in dec:
            words = line.split(" ")
            dec_len = len(words)
            self.dec_len = min(dec_len, self.dec_len)

        if len(enc) != len(dec):
            print(" Size error: {} ! ".format(name))

        return enc, dec, len(enc)

    def build_data(self):
        print(self.train_size, self.val_size, self.test_size)
        print(len(self.w2idx_enc), len(self.w2idx_dec))
        self.size = self.train_size + self.test_size + self.val_size
        self.train_idx = list(np.arange(0, self.train_size))
        self.val_idx = list(np.arange(self.train_size, self.train_size + self.val_size))
        self.test_idx = list(np.arange(self.size - self.test_size, self.size))

        enc_ids = np.zeros((self.size, self.enc_len), dtype=np.int32)
        dec_ids = np.zeros((self.size, self.dec_len), dtype=np.int32)
        length = np.zeros((self.size, 2), dtype=np.int32)

        for i in range(self.size):
            self.text_enc[i] += ' </s>'
            self.text_dec[i] += ' </s>'

            enc = self.tokenizer.tokenize(self.text_enc[i])
            dec = self.tokenizer.tokenize(self.text_dec[i])
            length[i][0] = min(len(enc), self.enc_len)
            length[i][1] = min(len(dec), self.dec_len)

            for j in range(length[i][0]):
                enc_ids[i][j] = self.word2idx(self.w2idx_enc, enc[j])
            for j in range(length[i][1]):
                dec_ids[i][j] = self.word2idx(self.w2idx_dec, dec[j])
        print(" | -> Reading Finish ")

        return enc_ids, dec_ids, length

    def get_batch(self, mode, size, idxs=None):
        if mode is 'train':
            idx = random.sample(self.train_idx, size)
            _idx = 0
        elif mode is 'val':
            idx = random.sample(self.val_idx, size)
            _idx = self.train_size
        else:
            idx = random.sample(self.test_idx, size)
            _idx = self.size - self.test_size

        if idxs is not None:
            idx = np.array(idxs) + _idx

        return idx, self.enc_ids[idx], self.dec_ids[idx], self.length[idx, 0], self.length[idx, 1]

    def save(self):
        print(' | Start saving ')
        total = {'w2idx_enc': self.w2idx_enc, 'w2idx_dec': self.w2idx_dec,
                 'text_enc': self.text_enc, 'text_dec': self.text_dec,
                 'enc_ids': self.enc_ids, 'dec_ids': self.dec_ids, 'length': self.length,
                 'train_idx': self.train_idx, 'val_idx': self.val_idx, 'test_idx': self.test_idx,
                 'enc_len': self.enc_len, 'enc_vocab': self.enc_vocab,
                 'dec_len': self.dec_len, 'dec_vocab': self.dec_vocab}
        np.save(self.save_name, total)
        print(' | -> Saving Finish ')

    def load(self):
        print(' | Load data: {}'.format(self.save_name))
        total = np.load(self.save_name).item()
        self.enc_ids, self.dec_ids, self.length = total['enc_ids'], total['dec_ids'], total['length']
        self.train_idx, self.val_idx, self.test_idx = total['train_idx'], total['val_idx'], total['test_idx']
        self.w2idx_enc, self.w2idx_dec = total['w2idx_enc'], total['w2idx_dec']
        self.text_enc, self.text_dec = total['text_enc'], total['text_dec']
        self.enc_len, self.enc_vocab = total['enc_len'], total['enc_vocab']
        self.dec_len, self.dec_vocab = total['dec_len'], total['dec_vocab']


