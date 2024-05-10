import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

param_train = np.load('../config/cpt_train.npy').transpose((0, 2, 1))
labels_train = np.load('../config/labels_train.npy')
area_train = np.load('../config/area_train.npy')
param_val = np.load('../config/cpt_val.npy').transpose((0, 2, 1))
labels_val = np.load('../config/labels_val.npy')
area_val = np.load('../config/area_val.npy')
param_test = np.load('../config/cpt_test.npy').transpose((0, 2, 1))
labels_test = np.load('../config/labels_test.npy')
area_test = np.load('../config/area_test.npy')

ind_train = area_train > 0
ind_val = area_val > 0
ind_test = area_test > 0
param_train = param_train[ind_train]
param_val = param_val[ind_val]
param_test = param_test[ind_test]
labels_train = labels_train[ind_train]
labels_val = labels_val[ind_val]
labels_test = labels_test[ind_test]
area_train = area_train[ind_train]
area_val = area_val[ind_val]
area_test = area_test[ind_test]


params_dataset = np.vstack([param_train, param_val, param_test])
labels_dataset = np.vstack([labels_train, labels_val, labels_test])
areas_dataset = np.hstack([area_train, area_val, area_test])
labels_dataset = np.hstack([labels_dataset, areas_dataset[:, np.newaxis]])

# params_dataset = np.vstack([data_test['params']])
# labels_dataset = np.vstack([data_test['labels']])

N_OUT = 3

lb = labels_dataset.min(0)
ub = labels_dataset.max(0)

BINS = 22
TEST_NUM = 10000

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
        for k in range(len(intervals) - 1):
            lbk2 = intervals[k, 2]
            ubk2 = intervals[k + 1, 2]
            flags.append(
                np.where(
                    (lbk2 <= labels_dataset[:, 2]) * (labels_dataset[:, 2] < ubk2) *
                    (lbj1 <= labels_dataset[:, 1]) * (labels_dataset[:, 1] < ubj1) *
                    (lbi0 <= labels_dataset[:, 0]) * (labels_dataset[:, 0] < ubi0)
                )[0]
            )

input('..')
N_flags = np.zeros(len(flags))
for i in range(len(flags)):
    N_flags[i] = len(flags[i])
# plt.matshow(N_flags.reshape(BINS, BINS))
# plt.show()

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
plt.hist(labels_test_set[:, 2], bins=100, color='green', density=True)
plt.hist(labels_dataset[:, 2], bins=100, color='red', alpha=0.5, density=True)
plt.show()

test_ind = np.array(test_ind)
total_ind = np.arange(len(labels_dataset))
train_val_ind = np.setdiff1d(total_ind, test_ind)
np.save('test_ind.npy', test_ind)
np.save('train_val_ind.npy', train_val_ind)


