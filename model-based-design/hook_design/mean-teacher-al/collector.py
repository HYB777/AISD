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

        # self.train_ind = np.setdiff1d(self.train_ind, add_ind[train_size:])
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
        # pool_ind_partial = self.pool_ind[:len(self.train_ind)]
        train_size = len(self.train_ind) + len(self.pool_ind)
        n_blocks = math.ceil(train_size/batch_size)
        assert len(self.train_ind) >= n_blocks and len(self.pool_ind) >= n_blocks, '#blocks are too much'
        labeled_blocks = np.array_split(self.train_ind, n_blocks)
        unlabeled_blocks = np.array_split(self.pool_ind, n_blocks)
        random.shuffle(labeled_blocks)
        random.shuffle(unlabeled_blocks)
        data_blocks = [np.hstack([a, b]) for a, b in zip(labeled_blocks, unlabeled_blocks)]
        teacher_noise_blocks = []
        student_noise_blocks = []
        random.shuffle(data_blocks)
        for data_block in data_blocks:
            np.random.shuffle(data_block)
            np.random.shuffle(self.pool_ind)
            teacher_noise_blocks.append(self.pool_ind[:len(data_block)])
            student_noise_blocks.append(self.pool_ind[-len(data_block):])
        return data_blocks, teacher_noise_blocks, student_noise_blocks

    # def train_blocks(self, batch_size):
    #     np.random.shuffle(self.train_ind)
    #     np.random.shuffle(self.pool_ind)
    #     pool_ind_partial = self.pool_ind[:len(self.train_ind)]
    #     pool_ind_residual = self.pool_ind[len(self.train_ind):]
    #     train_size = len(self.train_ind) + len(pool_ind_partial)
    #     n_blocks = math.ceil(train_size/batch_size)
    #     assert len(self.train_ind) >= n_blocks and len(pool_ind_partial) >= n_blocks, '#blocks are too much'
    #     labeled_blocks = np.array_split(self.train_ind, n_blocks)
    #     unlabeled_blocks = np.array_split(pool_ind_partial, n_blocks)
    #     random.shuffle(labeled_blocks)
    #     random.shuffle(unlabeled_blocks)
    #     data_blocks = [np.hstack([a, b]) for a, b in zip(labeled_blocks, unlabeled_blocks)]
    #     teacher_noise_blocks = []
    #     student_noise_blocks = []
    #     random.shuffle(data_blocks)
    #     for data_block in data_blocks:
    #         np.random.shuffle(data_block)
    #         np.random.shuffle(pool_ind_residual)
    #         teacher_noise_blocks.append(pool_ind_residual[:len(data_block)])
    #         student_noise_blocks.append(pool_ind_residual[-len(data_block):])
    #     return data_blocks, teacher_noise_blocks, student_noise_blocks

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

    def total(self):
        self.mode = 'total'

    def __getitem__(self, item):
        mode = self.mode.split('_')
        ind = self.ind_[item]
        is_labeled = self.is_labeled[ind]

        data, se, vol = self.dataset[ind]

        if mode[0] == 'pool':
            rand_ind = np.random.permutation(len(self.ind_))[:2]
            noise_ind_0 = self.ind_[rand_ind[0]]
            noise_ind_1 = self.ind_[rand_ind[1]]
            teacher_noise, _, _ = self.dataset[noise_ind_0]
            student_noise, _, _ = self.dataset[noise_ind_1]
            return data, ind, is_labeled, teacher_noise, student_noise
        else:
            return data, se, vol, is_labeled


class DataLoaderFit(Dataset):
    def __init__(self, params, labels):
        """
        :param params: (N, 11 + 11*5 + 8) [scales, sections_pts, guide_pts]
        :param labels: (N, 2)
        """
        super(DataLoaderFit, self).__init__()

        self.data = torch.from_numpy(params).float()
        self.se = torch.from_numpy(labels[:, 0]).float()
        self.vol = torch.from_numpy(labels[:, 1]).float()
        self.scale = torch.ones(2)

    def update_scale(self, data_ind):
        self.scale[0] = self.se[data_ind].std()
        self.scale[1] = self.vol[data_ind].std()
        return self.scale

    def __len__(self):
        return len(self.se)

    def __getitem__(self, item):
        data = self.data[item]
        se = self.se[item] / self.scale[0]
        vol = self.vol[item] / self.scale[1]

        return data, se, vol
