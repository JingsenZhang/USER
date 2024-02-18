import os
import math
import torch
import heapq
import random
import pickle
import numpy as np
import pandas as pd
import datetime


###############################################################################
# Utils
###############################################################################
def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']: '


def ids2tokens(ids, word2idx, idx2word):  # tranform ids to token_seq(sentence)
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens


def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


###############################################################################
# DataLoader
###############################################################################

class WordDictionary:  # word & feature
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']  # list
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}  # dict:{'<bos>':0, '<eos>':1, '<pad>':2, '<unk>':3}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):  # add word & record the word count
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:  # user & item
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class NewDataLoader:

    def __init__(self, data_path, train_path, valid_path, test_path, vocab_size):
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set()
        self.train, self.valid, self.test, self.user2items_train, self.user_inter_count = self.load_data(
            train_path,
            valid_path,
            test_path)
        self.train_size = len(self.train)
        self.valid_size = len(self.valid)
        self.test_size = len(self.test)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            # print('review',review)
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            (fea, adj, tem, sco) = review['template']
            self.word_dict.add_sentence(tem)
            self.word_dict.add_word(fea)  # add the feature into corpus
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, train_path, valid_path, test_path):
        train_data = pd.read_csv(train_path, header=0, names=['user', 'item', 'rating', 'text', 'feature', 'sco'],
                                 sep='\t')
        valid_data = pd.read_csv(valid_path, header=0, names=['user', 'item', 'rating', 'text', 'feature', 'sco'],
                                 sep='\t')
        test_data = pd.read_csv(test_path, header=0, names=['user', 'item', 'rating', 'text', 'feature', 'sco'],
                                sep='\t')

        train = train_data.to_dict('records')
        valid = valid_data.to_dict('records')
        test = test_data.to_dict('records')

        nuser = len(self.user_dict)
        user_inter_count = np.zeros(nuser)

        user2items_train = {}
        for x in train:
            u = x['user']
            i = x['item']
            f = x['feature']
            if u in user2items_train:
                user2items_train[u].add(i)
            else:
                user2items_train[u] = {i}
            user_inter_count[x['user']] += 1
            self.feature_set.add(f)

        for x in valid:
            f = x['feature']
            self.feature_set.add(f)

        for x in test:
            f = x['feature']
            self.feature_set.add(f)

        return train, valid, test, user2items_train, user_inter_count

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]


class PeterBatchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False):
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        u, i, r, t, f, s = [], [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append(sentence_format(eval(x['text']), seq_len, pad, bos, eos))
            f.append([x['feature']])
            s.append(x['sco'])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.score = torch.tensor(s, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        feature = self.feature[index]  # (batch_size, 1)
        score = self.score[index]  # (batch_size, 1)
        return user, item, rating, seq, feature, score, index
