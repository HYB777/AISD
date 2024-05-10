import numpy as np
import skdim


X = np.load('config/data_extend_wgan.npy').astype(np.float32)
alg = skdim.id.lPCA(ver='maxgap').fit_pw(X)
print(np.mean(alg.dimension_pw_))
