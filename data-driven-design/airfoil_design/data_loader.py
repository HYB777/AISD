import numpy as np
import torch

from torch.utils.data.dataset import Dataset


class DataLoaderFIT_CLDMP_DICT(Dataset):
    def __init__(self, data_dict, training):
        super(DataLoaderFIT_CLDMP_DICT, self).__init__()

        n = len(data_dict['airfoil'])
        self.data = torch.from_numpy(data_dict['airfoil']).reshape(n, 2, 32).float()
        self.alfa = torch.from_numpy(data_dict['res'][:, 0]).float()
        self.cl = torch.from_numpy(data_dict['res'][:, 1]).float()
        self.cd = torch.from_numpy(data_dict['res'][:, 2]).float()
        self.cm = torch.from_numpy(data_dict['res'][:, 3]).float()
        self.cp = torch.from_numpy(data_dict['res'][:, 4]).float()
        self.ind_map = data_dict['ind_map'].astype(np.int64)
        self.training = training

        print('total label: ', len(self.cl))

    def __len__(self):
        return len(self.cl)

    def __getitem__(self, item):
        data = self.data[self.ind_map[item]]
        alfa = self.alfa[item]
        if self.training:
            eps = torch.randn(1) * 0.05 / 3
            alfa += eps.item()
        cl = self.cl[item]
        cd = self.cd[item]
        cm = self.cm[item]
        cp = self.cp[item]

        return data, alfa, cl, cd, cm, cp


class DataLoaderAE(Dataset):
    def __init__(self, data_x):
        super(DataLoaderAE, self).__init__()
        n = len(data_x)
        self.data = data_x
        print('total label: ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        data_tensor = self.data[item]

        return data_tensor


class DataLoaderGAN(Dataset):
    def __init__(self, data_x):
        super(DataLoaderGAN, self).__init__()
        n = len(data_x)
        data_npy = data_x.reshape(n, 2, 32).numpy()
        data_npy[:, 1] = data_npy[:, 1][:, ::-1]
        self.data = torch.from_numpy(data_npy)
        print('total label: ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        data_tensor = self.data[item]

        return data_tensor
