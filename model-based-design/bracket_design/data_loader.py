import torch
from torch.utils.data.dataset import Dataset

class DataLoaderAE(Dataset):
    def __init__(self, params):
        super(DataLoaderAE, self).__init__()
        n = len(params)
        self.data = torch.from_numpy(params).float()
        print('total label: ', len(self.data), self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        data_tensor = self.data[item]
        return data_tensor

