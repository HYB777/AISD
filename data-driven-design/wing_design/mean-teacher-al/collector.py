import random
from copy import copy

import numpy as np
import torch
import math

from torch.utils.data.dataset import Dataset


class Collector(Dataset):
    def __init__(self, dataset, max_num=70_0000):
        super(Collector, self).__init__()
        self.dataset = dataset
        self.scales = dataset.scale

        self.total_ind = np.arange(len(self.dataset), dtype=np.int64)
        self.is_labeled = np.zeros_like(self.total_ind, dtype=np.int64)

        self.pool_ind = np.arange(len(self.dataset), dtype=np.int64)
        self.dataset_ind = np.array([], dtype=np.int64)
        self.train_ind = np.array([], dtype=np.int64)
        self.val_ind = np.array([], dtype=np.int64)

        self.max_num = max_num
        self.mode = 'pool'  # train_%d, eval_%d, pool, total
        self.pool_noise = False

        self.ind_ = None
        self.rand_pool_ind = None

        print('total label: ', len(self.dataset))

    def check_ok(self):
        assert len(np.intersect1d(self.train_ind, self.val_ind)) == 0, \
            'train set ^ val set != NULL'
        assert set(np.union1d(self.train_ind, self.val_ind)) == set(self.dataset_ind), \
            'train set U val set != labeled set'
        assert set(np.union1d(self.pool_ind, self.dataset_ind)) == set(self.total_ind), \
            'unlabeled set U labeled set != total set'
        assert np.sum(self.is_labeled) == len(self.dataset_ind), 'labeled set counting error'
        print('checking ok!')

    def data_size(self):
        return len(self.dataset_ind)

    def save_dataset_ind(self, ind_file):
        np.save(ind_file, self.dataset_ind)

    def state_dict(self):
        return {
            'is_labeled': self.is_labeled,
            'pool_ind': self.pool_ind,
            'dataset_ind': self.dataset_ind,
            'train_ind': self.train_ind,
            'val_ind': self.val_ind,
            'mode': self.mode,
            'scales': self.scales,
        }

    def load_state_dict(self, state_dict):
        self.is_labeled = state_dict['is_labeled']
        self.pool_ind = state_dict['pool_ind']
        self.dataset_ind = state_dict['dataset_ind']
        self.train_ind = state_dict['train_ind']
        self.val_ind = state_dict['val_ind']
        self.mode = state_dict['mode']
        self.scales = state_dict['scales']

    def reset(self):
        self.is_labeled = np.zeros_like(self.total_ind, dtype=np.int64)
        self.pool_ind = np.arange(len(self.dataset), dtype=np.int64)
        self.dataset_ind = np.array([], dtype=np.int64)
        self.train_ind = np.array([], dtype=np.int64)
        self.val_ind = np.array([], dtype=np.int64)
        self.mode = 'total'

    def terminated(self):
        return len(self.dataset_ind) >= self.max_num

    def random_add_from_pool(self, num=1000, split_ratio=10):
        np.random.shuffle(self.pool_ind)
        add_ind = self.pool_ind[:num]
        self.add_from_pool(add_ind, update_scales_flag=True, split_ratio=split_ratio)

    def add_from_pool(self, add_ind, update_scales_flag=True, split_ratio=10):
        self.is_labeled[add_ind] = 1

        self.dataset_ind = np.hstack([self.dataset_ind, add_ind])
        self.update_scale(update_scales_flag)

        self.pool_ind = np.setdiff1d(self.pool_ind, add_ind)
        self.split_train_val(add_ind, split_ratio)

    def split_train_val(self, add_ind, split_ratio):
        np.random.shuffle(add_ind)
        add_size = len(add_ind)

        if split_ratio > 0:
            train_size = add_size // split_ratio
            self.train_ind = np.hstack([self.train_ind, add_ind[train_size:]])
            self.val_ind = np.hstack([self.val_ind, add_ind[:train_size]])
        else:
            self.train_ind = np.hstack([self.train_ind, add_ind])
        self.check_ok()

    def update_scale(self, update_scales_flag):
        if update_scales_flag:
            self.scales = self.dataset.update_scale(self.dataset_ind)

    def __len__(self):
        mode = self.mode.split('_')
        if mode[0] == 'train':
            ind_ = self.train_ind
        elif mode[0] == 'eval':
            ind_ = self.val_ind
        elif mode[0] == 'total':
            ind_ = self.total_ind
        else:
            num = int(mode[1])
            if num > 0:
                return num
            else:
                ind_ = self.pool_ind
        return len(ind_)

    def train_blocks(self, batch_size):
        np.random.shuffle(self.train_ind)
        np.random.shuffle(self.pool_ind)
        pool_ind_temp = copy(self.pool_ind)[:2*len(self.train_ind)]
        np.random.shuffle(pool_ind_temp)
        train_size = len(self.train_ind) + len(pool_ind_temp)
        n_blocks = math.ceil(train_size/batch_size)
        assert len(self.train_ind) >= n_blocks and len(pool_ind_temp) >= n_blocks, '#blocks are too much'
        labeled_blocks = np.array_split(self.train_ind, n_blocks)
        unlabeled_blocks = np.array_split(pool_ind_temp, n_blocks)
        random.shuffle(labeled_blocks)
        random.shuffle(unlabeled_blocks)
        data_blocks = [np.hstack([a, b]) for a, b in zip(labeled_blocks, unlabeled_blocks)]
        teacher_noise_blocks = []
        student_noise_blocks = []
        random.shuffle(data_blocks)
        teacher_pool_ind = copy(self.pool_ind)
        student_pool_ind = copy(self.pool_ind)
        # np.random.shuffle(self.pool_ind)
        # teacher_pool_ind = np.hstack([teacher_pool_ind, self.pool_ind[:train_size - len(self.pool_ind)]])
        # np.random.shuffle(self.pool_ind)
        # student_pool_ind = np.hstack([student_pool_ind, self.pool_ind[:train_size - len(self.pool_ind)]])
        np.random.shuffle(teacher_pool_ind)
        np.random.shuffle(student_pool_ind)
        start = 0
        for data_block in data_blocks:
            # print(cnt, len(data_blocks), n_blocks, train_size, batch_size)
            np.random.shuffle(data_block)
            teacher_noise_blocks.append(teacher_pool_ind[start:start + len(data_block)])
            student_noise_blocks.append(student_pool_ind[start:start + len(data_block)])
            start = start + len(data_block)
        return data_blocks, teacher_noise_blocks, student_noise_blocks

    def train(self):
        self.mode = 'train'
        self.ind_ = copy(self.train_ind)

    def eval(self):
        self.mode = 'eval'
        self.ind_ = copy(self.val_ind)

    def pool(self, subset=-1, use_noise=False):
        self.mode = 'pool_%d' % subset
        mode = self.mode.split('_')
        if int(mode[1]) == -1:
            self.ind_ = copy(self.pool_ind)
        else:
            np.random.shuffle(self.pool_ind)
            self.ind_ = copy(self.pool_ind[:int(mode[1])])
        self.pool_noise = use_noise
        self.rand_pool_ind = np.random.permutation(len(self.pool_ind))

    def total(self):
        self.mode = 'total'

    def __getitem__(self, item):
        mode = self.mode.split('_')
        ind = self.ind_[item]
        is_labeled = self.is_labeled[ind]

        data, paras, cl, cd, vol = self.dataset[ind]

        if mode[0] == 'pool':
            # rand_ind = np.random.permutation(len(self.ind_))[:2]
            noise_ind_0 = self.ind_[self.rand_pool_ind[np.random.randint(len(self.rand_pool_ind))]]
            noise_ind_1 = self.ind_[self.rand_pool_ind[np.random.randint(len(self.rand_pool_ind))]]
            teacher_data_noise, teacher_paras_noise, _, _, _ = self.dataset[noise_ind_0]
            student_data_noise, student_paras_noise, _, _, _ = self.dataset[noise_ind_1]
            return data, paras, ind, is_labeled, teacher_data_noise, teacher_paras_noise, student_data_noise, student_paras_noise
        else:
            return data, paras, cl, cd, vol, is_labeled


class DataLoaderDICT_RMMTW(Dataset):
    def __init__(self, data_dict, base2, n_params):
        super(DataLoaderDICT_RMMTW, self).__init__()
        self.wings = torch.from_numpy(data_dict['wings']).float()
        res = torch.from_numpy(data_dict['res']).float()
        self.paras = res[:, :n_params]
        self.cl = res[:, n_params + 0]
        self.cd = res[:, n_params + 1]
        self.vol = res[:, n_params + 2]
        self.base2 = base2
        self.scale = torch.ones(3)

        print('total label: ', len(self.cl))

    def update_scale(self, data_ind):
        self.scale[0] = self.cl[data_ind].std()
        self.scale[1] = self.cd[data_ind].std()
        self.scale[2] = self.vol[data_ind].std()
        return self.scale

    def __len__(self):
        return len(self.cl)

    def __getitem__(self, item):
        data = self.wings[item // self.base2]
        paras = self.paras[item]

        cl = self.cl[item] / self.scale[0]
        cd = self.cd[item] / self.scale[1]
        vol = self.vol[item] / self.scale[2]

        return data, paras, cl, cd, vol
