import random
import numpy as np

class text_data(object):
    def __init__(self, path="../dataset/ptb", max_len=40, window=5, end_token="<eos>"):
        self.train_pt, self.val_pt, self.test_pt = 0, 0, 0
        self.train_start, self.val_start, self.test_start = 0, 0, 0
        self.train_end, self.val_end, self.test_end = 0, 0, 0

        self.path = path
        self.max_len = max_len
        self.window = window

        self.w2idx ={end_token: 0}
        self.train_text, self.train_pid, self.train_ids, self.train_target, self.train_plen = self.file_to_ids(path+"/ptb.train.txt")
        self.val_text, self.val_pid, self.val_ids, self.val_target, self.val_plen = self.file_to_ids(path + "/ptb.valid.txt")
        self.test_text, self.test_pid, self.test_ids, self.test_target, self.test_plen = self.file_to_ids(path + "/ptb.test.txt")
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

        all_ids = []
        text, paragraph_id, ids, target = [], [], [], []
        for num, line in enumerate(lines):
            text.append(line)
            id = np.zeros(40, dtype=np.int32)
            words = line.split()
            for i, word in enumerate(words):
                if i == self.max_len:
                    break
                if word not in self.w2idx:
                    self.w2idx[word] = len(self.w2idx)
                id[i] = self.w2idx[word]
            all_ids.append(id)

            if num == 5000:
                break

        p_length = np.zeros(len(all_ids))
        for num, line in enumerate(all_ids):
            for char_idx in range(len(line)):
                if char_idx+self.window+1 >= self.max_len or line[char_idx+self.window+1] == 0:
                    break
                else:
                    p_length[num] = p_length[num] + 1
                    paragraph_id.append(num)
                    ids.append(line[char_idx:char_idx+self.window])
                    target.append(line[char_idx+self.window+1])

        return text, np.array(paragraph_id), np.array(ids), np.array(target), p_length

    def get_train_all_text(self):
        return self.train_text

    def get_train_each_paragraph(self):
        pt = self.train_pt
        self.train_start = int(self.train_end)
        self.train_end = int(self.train_start + self.train_plen[pt])

        self.train_pt = self.train_pt+1
        if self.train_pt >= len(self.train_plen):
            self.train_pt, self.train_start, self.train_end = 0, 0, 0
        return self.train_pid[self.train_start:self.train_end], self.train_ids[self.train_start:self.train_end], self.train_target[self.train_start:self.train_end]
