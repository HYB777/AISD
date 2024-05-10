import os
import matplotlib.pyplot as plt
import torch

from nets import *
# import aerosandbox.numpy as np
from collector import DataLoaderDICT_RMMTW
from tqdm import tqdm
from copy import copy


# scales: cl: 2.6765e-03  cd: 4.3043e-03  vol: 2.2307e-03

n_params = 14


def _init_fn(worker_id):
    np.random.seed(1 + worker_id)


path = 'results-al-semi'
player_type = 'teacher'  # student, teacher, both


if player_type == 'teacher':
    players = ['teacher']
elif player_type == 'student':
    players = ['student']
else:
    players = ['teacher', 'student']

nets = []
N_nets = len(players)
check = torch.load('%s/net_best_teacher.tar' % path, map_location=torch.device('cpu'))
mu_paras = check['option']['mu_paras']
sig_paras = check['option']['sig_paras']

for player in players:
    check_i = torch.load('%s/net_best_%s.tar' % (path, player), map_location=torch.device('cpu'))
    # check_i = torch.load('%s/net_last.tar' % (path, ))
    net_i = ResNet1d18_RMMTW(n_params)
    net_i.load_state_dict(check_i[player])
    net_i.cpu()
    net_i.eval()
    nets.append(copy(net_i))

data_train_1 = np.load('../config/data_train_wings_rmmtw_new_0_100000.npy', allow_pickle=True).item()
data_train_2 = np.load('../config/data_train_wings_rmmtw_new_100000_200000.npy', allow_pickle=True).item()
data_train_3 = np.load('../config/data_train_wings_rmmtw_new_200000_300000.npy', allow_pickle=True).item()
data_train_4 = np.load('../config/data_train_wings_rmmtw_new_300000_400000.npy', allow_pickle=True).item()
data_train_5 = np.load('../config/data_train_wings_rmmtw_new_400000_500000.npy', allow_pickle=True).item()


data_train = {}
for k in data_train_1.keys():
    data_train[k] = np.vstack([data_train_1[k], data_train_2[k], data_train_3[k], data_train_4[k], data_train_5[k]])
test_ind = np.load('test_ind.npy')
data_train['res'] = data_train['res'].reshape((-1, 64, n_params + 3))
for k in data_train.keys():
    data_train[k] = data_train[k][test_ind]
data_train['res'] = data_train['res'].reshape((-1, n_params + 3))

# mu = data_train['params'].mean(0)
# sig = data_train['params'].std(0)
# params_test = (params_test - mu) / sig

scales = check['scales']  # torch.ones_like(check['scales'])
scale_cl = scales[0].item()
scale_cd = scales[1].item()
scale_vol = scales[2].item()

data_train['res'][:, :n_params] = (data_train['res'][:, :n_params] - mu_paras) / sig_paras
data_train['res'][:, n_params + 0] = data_train['res'][:, n_params + 0] / scale_cl
data_train['res'][:, n_params + 1] = data_train['res'][:, n_params + 1] / scale_cd
data_train['res'][:, n_params + 2] = data_train['res'][:, n_params + 2] / scale_vol

# test_ind = np.load('test_ind.npy')
# param_test = param_test[test_ind]
# labels_test = labels_test[test_ind]

test_set = DataLoaderDICT_RMMTW(data_train, 64, n_params)
# test_set.scale = scales
test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=4,
                                          batch_size=1000, shuffle=False, drop_last=False,
                                          pin_memory=False, worker_init_fn=_init_fn)

fit_loss = nn.L1Loss()


def nets_(X, paras):
    z1 = 0
    z2 = 0
    z3 = 0
    for i in range(N_nets):
        z123_ = nets[i](X, paras)
        z1 += z123_[:, 0]
        z2 += z123_[:, 1]
        z3 += z123_[:, 2]
    return z1 / N_nets, z2 / N_nets, z3 / N_nets


with torch.no_grad():
    # check = torch.load('%s/net_best.tar' % self.opt['checkpoints'])

    test_bar = tqdm(test_loader)
    total_loss = 0
    vol_rmae = 0
    mass_rmae = 0

    rmae1_u = 0
    rmae2_u = 0
    rmae3_u = 0
    rmae1_d = 0
    rmae2_d = 0
    rmae3_d = 0

    for X, paras, cl, cd, vol in test_bar:

        z1, z2, z3 = nets_(X, paras)

        loss = fit_loss(z1.reshape(-1), cl.reshape(-1)) + \
               fit_loss(z2.reshape(-1), cd.reshape(-1)) + \
               fit_loss(z3.reshape(-1), vol.reshape(-1))
        rmae1 = (torch.abs(cl.reshape(-1) - z1.reshape(-1)).sum() / torch.abs(cl).sum()).item()
        rmae2 = (torch.abs(cd.reshape(-1) - z2.reshape(-1)).sum() / torch.abs(cd).sum()).item()
        rmae3 = (torch.abs(vol.reshape(-1) - z3.reshape(-1)).sum() / torch.abs(vol).sum()).item()

        rmae1_u += torch.abs(cl.reshape(-1) - z1.reshape(-1)).sum().item()
        rmae1_d += torch.abs(cl).sum().item()
        rmae2_u += torch.abs(cd.reshape(-1) - z2.reshape(-1)).sum().item()
        rmae2_d += torch.abs(cd).sum().item()
        rmae3_u += torch.abs(vol.reshape(-1) - z3.reshape(-1)).sum().item()
        rmae3_d += torch.abs(vol).sum().item()

        total_loss += loss.item()

        test_bar.set_description(desc='loss: %.4e,  cl: %.4e,  cd: %.4e, vol: %.4e'
                                      % (loss.item(), rmae1, rmae2, rmae3))

    print('testing phase: loss: %.4e  cl: %.4e  cd: %.4e  vol: %.4e'
          % (total_loss / len(test_loader),
             rmae1_u / rmae1_d,
             rmae2_u / rmae2_d,
             rmae3_u / rmae3_d))
