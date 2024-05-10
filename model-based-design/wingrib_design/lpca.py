import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from annoy import AnnoyIndex

from tqdm import tqdm
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import skdim

param_train = np.load('data/holes_cpt_train.npy')
param_val = np.load('data/holes_cpt_val.npy')
param_test = np.load('data/holes_cpt_test.npy')
minn = param_train.min((0, 1, -1))
maxx = param_train.max((0, 1, -1))
mu = (maxx + minn) / 2
sig = (maxx - minn) / 2
mu = mu.reshape(1, 1, 3, 2, 1)
sig = sig.reshape(1, 1, 3, 2, 1)

param_train = (param_train - mu) / sig
param_val = (param_val - mu) / sig
param_test = (param_test - mu) / sig

X = np.vstack([param_train, param_val, param_test])  # [:, 66:]
b = X.shape[0]
x0 = X[:, :, 0].reshape(b * 3, 2, -1)
x1 = X[:, :, 1].reshape(b * 3, 2, -1)
x2 = X[:, :, 2].reshape(b * 3, 2, -1)
X = np.hstack([x0.reshape(b * 3, -1), x1.reshape(b * 3, -1), x2.reshape(b * 3, -1)])
per = np.random.permutation(X.shape[0])

alg = skdim.id.lPCA(ver='maxgap').fit_pw(X)
print(np.mean(alg.dimension_pw_))
