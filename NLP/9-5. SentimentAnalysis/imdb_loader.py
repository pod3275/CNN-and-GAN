import os
import random
import numpy as np

class text_data(object):
    def __init__(self, path="./dataset/aclImdb/", max_vocab=20000, max_len=100, end_token="<eos>"):
        self.train_pt, self.val_pt, self.test_pt = 0, 0, 0
        self.path = path
        self.max_len = max_len
        self.max_vocab = max_vocab

        self.w2idx = {end_token: 0, "<unk>": 1}
        self.train_ids, self.train_len, self.train_label = self.files_to_ids(path + "train/")
        self.test_ids, self.test_len, self.test_label = self.files_to_ids(path + "test/")
        self.vocab_size = len(self.w2idx)

        self.train_size = len(self.train_ids)
        self.test_size = len(self.test_ids)

        self.idx2w = {}
        for word in self.w2idx:
            self.idx2w[self.w2idx[word]] = word

    def get_w2idx(self, word):
        return 1 if word not in self.w2idx else self.w2idx[word]

    def files_to_ids(self, path):
        pos_list = os.listdir(path + "/pos")
        neg_list = os.listdir(path + "/neg")

        size = len(pos_list)
        lines = []
        for i in range(size):
            with open(path + "/neg/" + neg_list[i], "r", encoding="utf-8") as fin:
                lines.append(fin.readline())
            with open(path + "/pos/" + pos_list[i], "r", encoding="utf-8") as fin:
                lines.append(fin.readline())

        if "train" in path:
            cnt = {}
            for line in lines:
                for word in line.split():
                    if word in cnt:
                        cnt[word] += 1
                    else:
                        cnt[word] = 1
            cnt_sort = sorted(cnt.items(), key=lambda cnt:cnt[1], reverse=True)
            for word, count in cnt_sort:
                self.w2idx[word] = len(self.w2idx)
                if self.w2idx == self.max_vocab:
                    break

        length, ids, label = [], [], []
        for num, line in enumerate(lines):
            id = np.zeros(self.max_len, dtype=np.int32)
            line += " <eos>"
            words = line.split()
            for i, word in enumerate(words):
                if i == self.max_len:
                    break
                if word not in self.w2idx and len(self.w2idx) < self.max_vocab:
                    self.w2idx[word] = len(self.w2idx)
                id[i] = self.get_w2idx(word)
            ids.append(id)
            length.append(i)
            label.append(num % 2)

        return np.array(ids), np.array(length), np.array(label)

    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % self.train_size
        return self.train_ids[pt: pt+batch_size], self.train_len[pt: pt+batch_size], self.train_label[pt: pt+batch_size]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % self.test_size
        return self.test_ids[pt: pt+batch_size], self.test_len[pt: pt+batch_size], self.test_label[pt: pt+batch_size]
