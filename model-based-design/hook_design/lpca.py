import numpy as np
import skdim

X0 = np.load('data/data_train.npy', allow_pickle=True).item()['params']
X1 = np.load('data/data_val.npy', allow_pickle=True).item()['params']
X2 = np.load('data/data_test.npy', allow_pickle=True).item()['params']
X = np.vstack([X0, X1, X2])  # [:, 66:]
print(X.shape)

alg = skdim.id.lPCA(ver='maxgap').fit_pw(X)
print( np.mean(alg.dimension_pw_))
