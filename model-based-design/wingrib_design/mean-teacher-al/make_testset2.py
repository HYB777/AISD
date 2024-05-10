import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

param_train = np.load('../data/holes_cpt_train.npy')
labels_train = np.load('../data/labels_train.npy')
param_val = np.load('../data/holes_cpt_val.npy')
labels_val = np.load('../data/labels_val.npy')
param_test = np.load('../data/holes_cpt_test.npy')
labels_test = np.load('../data/labels_test.npy')


params_dataset = np.vstack([param_train, param_val, param_test])
labels_dataset = np.vstack([labels_train, labels_val, labels_test])

# params_dataset = np.vstack([data_test['params']])
# labels_dataset = np.vstack([data_test['labels']])

N_OUT = 2

lb = labels_dataset.min(0)
ub = labels_dataset.max(0)

BINS = 100
TEST_NUM = 20000

intervals = np.linspace(lb, ub, BINS + 1)

# flag = [[]] * N_OUT
# for i in range(N_OUT):
#     for j in range(len(intervals)-1):
#         lbj = intervals[j, i]
#         ubj = intervals[j + 1, i]
#         flag[i].append((lbj <= labels_dataset[:, i]) * (labels_dataset[:, i] < ubj))


flags = []
for i in range(len(intervals) - 1):
    lbi0 = intervals[i, 0]
    ubi0 = intervals[i + 1, 0]
    for j in range(len(intervals) - 1):
        lbj1 = intervals[j, 1]
        ubj1 = intervals[j + 1, 1]
        flags.append(np.where((lbj1 <= labels_dataset[:, 1]) * (labels_dataset[:, 1] < ubj1) * (lbi0 <= labels_dataset[:, 0]) * (labels_dataset[:, 0] < ubi0))[0])

input('..')
N_flags = np.zeros(len(flags))
for i in range(len(flags)):
    N_flags[i] = len(flags[i])
plt.matshow(N_flags.reshape(BINS, BINS))
plt.show()

test_ind = []

while len(test_ind) < TEST_NUM:
    for i in tqdm(range(len(flags))):
        if N_flags[i] > 0:
            np.random.shuffle(flags[i])
            test_ind.append(flags[i][0])
            flags[i] = flags[i][1:]
            N_flags[i] -= 1

pool_ind = np.random.permutation(len(labels_dataset))
labels_random = labels_dataset[pool_ind[:TEST_NUM]]

labels_test_set = labels_dataset[test_ind]
plt.hist(labels_test_set[:, 0], bins=100, color='green', density=True)
plt.hist(labels_dataset[:, 0], bins=100, color='red', alpha=0.5, density=True)
plt.show()
plt.hist(labels_test_set[:, 1], bins=100, color='green', density=True)
plt.hist(labels_dataset[:, 1], bins=100, color='red', alpha=0.5, density=True)
plt.show()

# test_ind = np.array(test_ind)
# total_ind = np.arange(len(labels_dataset))
# train_val_ind = np.setdiff1d(total_ind, test_ind)
# np.save('test_ind.npy', test_ind)
# np.save('train_val_ind.npy', train_val_ind)


