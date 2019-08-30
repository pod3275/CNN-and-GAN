import os
import random
import numpy as np


class QAdata(object):
    def __init__(self, path="./dataset/babi/",
                 max_q_len=20, max_t_len=20):

        self.path = path
        self.w2idx = {"<eos>": 0}
        self.max_q_len = max_q_len
        self.max_t_len = max_t_len

        self.T = np.zeros((0, self.max_t_len), dtype=np.int32)
        self.Q = np.zeros((0, self.max_q_len), dtype=np.int32)
        self.A = np.zeros((0), dtype=np.int32)
        self.t_len = np.zeros((0), dtype=np.int32)
        self.q_len = np.zeros((0), dtype=np.int32)

        self.read_qa(path + "/qa1_single-supporting-fact_train.txt")
        self.read_qa(path + "/qa2_two-supporting-facts_train.txt")

        self.total_size = len(self.T)
        self.idx = np.array((list(np.arange(0, self.total_size))))
        random.shuffle(self.idx)
        self.train_pt, self.test_pt = 0, int(self.total_size * 0.9)

        self.vocab_size = len(self.w2idx)
        self.idx2w = {}
        for word in self.w2idx:
            self.idx2w[self.w2idx[word]] = word

    def ids2sent(self, tokens):
        if type(tokens)==int:
            return self.idx2w[tokens]
        else:
            line = " "
            for t in tokens:
                line += self.idx2w[t] + " "
                if t == 0:
                    break
            return line

    def read_qa(self, file_name):
        with open(file_name, "r") as fin:
            lines = fin.readlines()

        Text, Qst, ans, t_len, q_len = [], [], [], [], []
        text = []
        for line in lines:
            tabs = line.split("\t")
            words = line.split()

            if int(words[0]) == 1:
                text = []
            else:
                text = list(text)

            # text
            if len(tabs) < 2:
                for i in range(1, len(words)):
                    word = words[i]
                    if word not in self.w2idx:
                        self.w2idx[word] = len(self.w2idx)
                    text.append(self.w2idx[word])

            else:
                q = []
                for i in range(1, len(tabs[0].split())):
                    word = words[i]
                    if word not in self.w2idx:
                        self.w2idx[word] = len(self.w2idx)
                    q.append(self.w2idx[word])

                answer = tabs[1].split()[0]
                if answer not in self.w2idx:
                    self.w2idx[answer] = len(self.w2idx)

                if len(text) <= self.max_t_len and len(q) <= self.max_q_len:
                    np_text = np.zeros((1, self.max_t_len), dtype=np.int32)
                    for i, o in enumerate(text):
                        np_text[0][i] = o

                    np_qst = np.zeros((1, self.max_q_len), dtype=np.int32)
                    for i, o in enumerate(q):
                        np_qst[0][i] = o

                    self.T = np.append(self.T, np_text, axis=0)
                    self.Q = np.append(self.Q, np_qst, axis=0)
                    self.A = np.append(self.A, self.w2idx[answer])
                    self.t_len = np.append(self.t_len, len(text))
                    self.q_len = np.append(self.q_len, len(q))


    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % int(self.total_size * 0.9)
        idxs = self.idx[pt: pt+batch_size]
        return self.T[idxs], self.Q[idxs], self.A[idxs], self.t_len[idxs], self.q_len[idxs]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % int(self.total_size * 0.1) + int(self.total_size * 0.9)
        idxs = self.idx[pt: pt+batch_size]
        return self.T[idxs], self.Q[idxs], self.A[idxs], self.t_len[idxs], self.q_len[idxs]
        # return self.T[pt: pt + batch_size], self.Q[pt: pt + batch_size], self.A[pt: pt + batch_size], self.t_len[pt: pt + batch_size], self.q_len[pt: pt + batch_size]
