import os
import matplotlib.pyplot as plt
import torch

from nets import *
# import aerosandbox.numpy as np
from collector import DataLoaderFit
from tqdm import tqdm
from copy import copy


def _init_fn(worker_id):
    np.random.seed(1 + worker_id)


path = 'results-al-semi' 
player_type = 'teacher' 


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
    net_i = HookPhyNetFC(74)
    net_i.load_state_dict(check_i[player])
    net_i.cpu()
    net_i.eval()
    nets.append(copy(net_i))

data_train = np.load('../data/data_train.npy', allow_pickle=True).item()
data_val = np.load('../data/data_val.npy', allow_pickle=True).item()
data_test = np.load('../data/data_test.npy', allow_pickle=True).item()

params_dataset = np.vstack([data_train['params'], data_val['params'], data_test['params']])
labels_dataset = np.vstack([data_train['labels'], data_val['labels'], data_test['labels']])
test_ind = np.load('test_ind.npy')
params_test = params_dataset[test_ind]
labels_tesst = labels_dataset[test_ind]

# mu = data_train['params'].mean(0)
# sig = data_train['params'].std(0)
# params_test = (params_test - mu) / sig

scales = check['scales']
scale_se = scales[0].item()
scale_vol = scales[1].item()

param_test = (params_test - mu_paras) / sig_paras
labels_tesst[:, 0] /= scale_se
labels_tesst[:, 1] /= scale_vol

# test_ind = np.load('test_ind.npy')
# param_test = param_test[test_ind]
# labels_test = labels_test[test_ind]

test_set = DataLoaderFit(param_test, labels_tesst)
test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=4,
                                          batch_size=128, shuffle=False, drop_last=False,
                                          pin_memory=False, worker_init_fn=_init_fn)

fit_loss = nn.L1Loss()


def nets_(X):
    z1 = 0
    z2 = 0
    for i in range(N_nets):
        z12_ = nets[i](X)
        z1 += z12_[:, 0]
        z2 += z12_[:, 1]
    return z1 / N_nets, z2 / N_nets


with torch.no_grad():
    # check = torch.load('%s/net_best.tar' % self.opt['checkpoints'])

    test_bar = tqdm(test_loader)
    total_loss = 0
    vol_rmae = 0
    mass_rmae = 0

    rmae1_u = 0
    rmae2_u = 0
    rmae1_d = 0
    rmae2_d = 0
    rmae1s = 0.
    rmae2s = 0.
    N = len(test_set)

    for X, vm, mass in test_bar:

        # X = X.cuda()
        # vm = vm.cuda()
        # mass = mass.cuda()

        z1, z2 = nets_(X)

        loss = fit_loss(z1.reshape(-1), vm.reshape(-1)) + \
               fit_loss(z2.reshape(-1), mass.reshape(-1))
        rmae1 = (torch.abs(vm.reshape(-1) - z1.reshape(-1)).sum() / torch.abs(vm).sum()).item()
        rmae2 = (torch.abs(mass.reshape(-1) - z2.reshape(-1)).sum() / torch.abs(mass).sum()).item()
        rmae1_u += torch.abs(vm.reshape(-1) - z1.reshape(-1)).sum().item()
        rmae1_d += torch.abs(vm).sum().item()
        rmae2_u += torch.abs(mass.reshape(-1) - z2.reshape(-1)).sum().item()
        rmae2_d += torch.abs(mass).sum().item()
        rmae1s += (torch.abs(vm.reshape(-1) - z1.reshape(-1))/torch.abs(vm).reshape(-1)).sum()
        rmae2s += (torch.abs(mass.reshape(-1) - z2.reshape(-1)) / torch.abs(mass).reshape(-1)).sum()

        total_loss += loss.item()
        vol_rmae += rmae1
        mass_rmae += rmae2

        test_bar.set_description(desc='loss: %.4e,  se: %.4e,  vol: %.4e'
                                      % (loss.item(), rmae1, rmae2))

    print('testing phase: loss: %.4e  se: %.4e  vol: %.4e'
          % (total_loss / len(test_loader),
             rmae1s/N,
             rmae2s/N))
    print(rmae1s/N, rmae2s/N)

    # rmae1_u / rmae1_d,.
    # rmae2_u / rmae2_d))
