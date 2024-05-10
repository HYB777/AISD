import numpy as np
import skdim

param_train = np.load('config/cpt_train.npy').transpose((0, 2, 1))
param_val = np.load('config/cpt_val.npy').transpose((0, 2, 1))
param_test = np.load('config/cpt_test.npy').transpose((0, 2, 1))
area_train = np.load('config/area_train.npy')
area_val = np.load('config/area_val.npy')
area_test = np.load('config/area_test.npy')
ind_train = area_train > 0
ind_val = area_val > 0
ind_test = area_test > 0

param_train = param_train[ind_train]
param_val = param_val[ind_val]
param_test = param_test[ind_test]
X = np.vstack([param_train, param_val, param_test]).reshape(-1, 64)

print(X.shape)

alg = skdim.id.lPCA(ver='maxgap').fit_pw(X)
print(np.mean(alg.dimension_pw_))
