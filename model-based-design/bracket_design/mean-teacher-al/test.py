import os
import matplotlib.pyplot as plt
import torch

from nets import *
import numpy as np
from collector import DataLoaderFIT
from tqdm import tqdm
from copy import copy


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
    net_i = ResNet1d18Fit()
    net_i.load_state_dict(check_i[player])
    net_i.cpu()
    net_i.eval()
    nets.append(copy(net_i))

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

test_ind = np.load('test_ind.npy')
params_test = params_dataset[test_ind]
labels_test = labels_dataset[test_ind]
areas_test = areas_dataset[test_ind]

# mu = data_train['params'].mean(0)
# sig = data_train['params'].std(0)
# params_test = (params_test - mu) / sig

scales = check['scales']
scale_freq = scales[0].item()
scale_vm = scales[1].item()
scale_area = scales[2].item()

param_test = (params_test - mu_paras) / sig_paras
labels_test[:, 0] /= scale_freq
labels_test[:, 1] /= scale_vm
areas_test /= scale_area

# test_ind = np.load('test_ind.npy')
# param_test = param_test[test_ind]
# labels_test = labels_test[test_ind]

test_set = DataLoaderFIT(param_test, labels_test, areas_test)
test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=4,
                                          batch_size=1000, shuffle=False, drop_last=False,
                                          pin_memory=False, worker_init_fn=_init_fn)

fit_loss = nn.L1Loss()


def nets_(X):
    z1 = 0
    z2 = 0
    z3 = 0
    for i in range(N_nets):
        z123_ = nets[i](X)
        z1 += z123_[:, 0]
        z2 += z123_[:, 1]
        z3 += z123_[:, 2]
    return z1 / N_nets, z2 / N_nets, z3 / N_nets


with torch.no_grad():
    # check = torch.load('%s/net_best.tar' % self.opt['checkpoints'])

    test_bar = tqdm(test_loader)
    total_loss = 0
    freq_rmae = 0
    vm_rmae = 0
    area_rmae = 0

    rmae1_u = 0
    rmae2_u = 0
    rmae3_u = 0
    rmae1_d = 0
    rmae2_d = 0
    rmae3_d = 0

    rmae1s = 0.
    rmae2s = 0.
    rmae3s = 0.
    N = len(test_set)

    for X, freq, vm, area in test_bar:

        # X = X.cuda()
        # vm = vm.cuda()
        # mass = mass.cuda()

        z1, z2, z3 = nets_(X)

        loss = fit_loss(z1.reshape(-1), freq.reshape(-1)) + \
               fit_loss(z2.reshape(-1), vm.reshape(-1)) + \
               fit_loss(z3.reshape(-1), area.reshape(-1))
        rmae1 = (torch.abs(freq.reshape(-1) - z1.reshape(-1)).sum() / torch.abs(freq).sum()).item()
        rmae2 = (torch.abs(vm.reshape(-1) - z2.reshape(-1)).sum() / torch.abs(vm).sum()).item()
        rmae3 = (torch.abs(area.reshape(-1) - z3.reshape(-1)).sum() / torch.abs(area).sum()).item()
        rmae1_u += torch.abs(freq.reshape(-1) - z1.reshape(-1)).sum().item()
        rmae1_d += torch.abs(freq).sum().item()
        rmae2_u += torch.abs(vm.reshape(-1) - z2.reshape(-1)).sum().item()
        rmae2_d += torch.abs(vm).sum().item()
        rmae3_u += torch.abs(area.reshape(-1) - z3.reshape(-1)).sum().item()
        rmae3_d += torch.abs(area).sum().item()

        rmae1s += (torch.abs(freq.reshape(-1) - z1.reshape(-1))/torch.abs(freq.reshape(-1))).sum()
        rmae2s += (torch.abs(vm.reshape(-1) - z2.reshape(-1)) / torch.abs(vm.reshape(-1))).sum()
        rmae3s += (torch.abs(area.reshape(-1) - z3.reshape(-1)) / torch.abs(area.reshape(-1))).sum()

        total_loss += loss.item()

        test_bar.set_description(desc='loss: %.4e,  freq: %.4e,  vm: %.4e,  area: %.4e'
                                      % (loss.item(), rmae1, rmae2, rmae3))

    print('testing phase: loss: %.4e  freq: %.4e  vm: %.4e  area: %.4e'
          % (total_loss / len(test_loader),
             rmae1s / N, rmae2s / N, rmae3s / N
             )
          )
    print(rmae1s/N, rmae2s/N, rmae3s/N)

    # rmae1_u / rmae1_d,
    # rmae2_u / rmae2_d,
    # rmae3_u / rmae3_d,
