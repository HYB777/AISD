import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


data_train = np.load('../data/data_train.npy', allow_pickle=True).item()
data_val = np.load('../data/data_val.npy', allow_pickle=True).item()
data_test = np.load('../data/data_test.npy', allow_pickle=True).item()
params_dataset = np.vstack([data_train['params'], data_val['params'], data_test['params']])
labels_dataset = np.vstack([data_train['labels'], data_val['labels'], data_test['labels']])


test_ind = np.load('test_ind.npy')
al_ind = torch.load('results-al-1e-4/checkpoints_last.tar')['collector']['dataset_ind']
random_ind = torch.load('results-random-1e-4/checkpoints_last.tar')['collector']['dataset_ind']

for i in range(labels_dataset.shape[1]):
    plt.hist(labels_dataset[test_ind, i], bins=100, color='green', density=True)
    plt.hist(labels_dataset[al_ind, i], bins=100, color='red', density=True, alpha=0.7)
    plt.hist(labels_dataset[random_ind, i], bins=100, color='blue', density=True, alpha=0.5)
    plt.show()