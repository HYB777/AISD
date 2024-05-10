import numpy as np
import torch
import scipy.io as scio
from torch.utils.data.dataset import Dataset


def mat2npy(mat_str_list):
    mat_list = [scio.loadmat(it) for it in mat_str_list]
    scales = np.vstack([it['scales'] if 'scales' in it.keys() else it['scales_i'] for it in mat_list])
    sections_pts = np.vstack([it['section_pts'].reshape(-1, 5*10) if 'section_pts' in it.keys() else it['section_pts_i'].reshape(-1, 5*10) for it in mat_list])
    guide_pts = np.vstack([it['guide_pts'] if 'guide_pts' in it.keys() else it['guide_pts_i'] for it in mat_list])
    labels = np.vstack([it['results'] if 'results' in it.keys() else it['results_i'] for it in mat_list])
    params = np.hstack([scales, sections_pts, guide_pts])
    return params, labels


class DataLoaderFit(Dataset):
    def __init__(self, params, labels):
        """
        :param params: (N, 11 + 11*5 + 8) [scales, sections_pts, guide_pts]
        :param labels: (N, 2)
        """
        super(DataLoaderFit, self).__init__()

        self.data = torch.from_numpy(params).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.labels[item]

        return data, label


class DataLoaderAE(Dataset):
    def __init__(self, params):
        """
        :param params: (N, 11 + 11*5 + 8) [scales, sections_pts, guide_pts]
        """
        super(DataLoaderAE, self).__init__()

        self.data = torch.from_numpy(params).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        data_tensor = self.data[item]

        return data_tensor


