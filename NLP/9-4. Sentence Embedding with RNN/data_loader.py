import random
import numpy as np

class text_data(object):
    def __init__(self, path="../dataset/ptb", max_len=40, end_token="<eos>"):
        self.train_pt, self.val_pt, self.test_pt = 0, 0, 0
        self.path = path
        self.max_len = max_len

        self.w2idx ={end_token: 0}
        self.train_text, self.train_ids, self.train_len = self.file_to_ids(path+"/ptb.train.txt")
        self.val_text, self.val_ids, self.val_len = self.file_to_ids(path + "/ptb.valid.txt")
        self.test_text, self.test_ids, self.test_len = self.file_to_ids(path + "/ptb.test.txt")
        self.vocab_size = len(self.w2idx)

        self.train_size = len(self.train_ids)
        self.val_size = len(self.val_ids)
        self.test_size = len(self.test_ids)

        self.idx2w = {}
        for word in self.w2idx:
            self.idx2w[self.w2idx[word]] = word

    def file_to_ids(self, file_name):
        with open(file_name, "r") as fin:
            lines = fin.readlines()

        text, length, ids = [], [], []
        for num, line in enumerate(lines):
            text.append(line)
            id = np.zeros(self.max_len, dtype=np.int32)
            line += " <eos>"
            words = line.split()
            for i, word in enumerate(words):
                if i == self.max_len:
                    break
                if word not in self.w2idx:
                    self.w2idx[word] = len(self.w2idx)
                id[i] = self.w2idx[word]
            ids.append(id)
            length.append(i)

            if num == 100000:
                break

        return text, np.array(ids), np.array(length)

    def get_train_all(self):
        return self.train_text, self.train_ids, self.train_len

    def get_train(self, batch_size=20):
        pt = self.train_pt
        self.train_pt = (self.train_pt + batch_size) % self.train_size
        return self.train_ids[pt: pt+batch_size], self.train_len[pt: pt+batch_size]

    def get_val(self, batch_size=20):
        pt = self.val_pt
        self.val_pt = (self.val_pt + batch_size) % self.val_size
        return self.val_ids[pt: pt+batch_size], self.val_len[pt: pt+batch_size]

    def get_test(self, batch_size=20):
        pt = self.test_pt
        self.test_pt = (self.test_pt + batch_size) % self.test_size
        return self.test_ids[pt: pt+batch_size], self.test_len[pt: pt+batch_size]
