import os
import matplotlib.pyplot as plt
from nets import *
import numpy as np
from collector import DataLoaderFIT
from tqdm import tqdm
from copy import copy


def _init_fn(worker_id):
    np.random.seed(1 + worker_id)


path = './results-al-semi'
player_type = 'teacher'  # student, teacher, both


if player_type == 'teacher':
    players = ['teacher']
elif player_type == 'student':
    players = ['student']
else:
    players = ['teacher', 'student']

nets = []
N_nets = len(players)
check = torch.load('%s/net_best_teacher.tar' % path)
mu_paras = check['option']['mu_paras']
sig_paras = check['option']['sig_paras']
for player in players:
    check_i = torch.load('%s/net_best_%s.tar' % (path, player))
    # check_i = torch.load('%s/net_last.tar' % (path, ))
    net_i = ResNet1d18Fit()
    net_i.load_state_dict(check_i[player])
    net_i.cuda()
    net_i.eval()
    nets.append(copy(net_i))

param_train = np.load('../data/holes_cpt_train.npy')
labels_train = np.load('../data/labels_train.npy')
param_val = np.load('../data/holes_cpt_val.npy')
labels_val = np.load('../data/labels_val.npy')
param_test = np.load('../data/holes_cpt_test.npy')
labels_test = np.load('../data/labels_test.npy')

params_dataset = np.vstack([param_train, param_val, param_test])
labels_dataset = np.vstack([labels_train, labels_val, labels_test])

test_ind = np.load('test_ind.npy')
params_test = params_dataset[test_ind]
labels_test = labels_dataset[test_ind]

scales = check['scales']
scale_vm = scales[0].item()
scale_mass = scales[1].item()

param_test = (params_test - mu_paras) / sig_paras
labels_test[:, 0] /= scale_vm
labels_test[:, 1] /= scale_mass

test_set = DataLoaderFIT(param_test, labels_test)
test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=4,
                                          batch_size=2083, shuffle=False, drop_last=False,
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
    vm_rmae = 0
    mass_rmae = 0

    rmae1_u = 0
    rmae2_u = 0
    rmae1_d = 0
    rmae2_d = 0

    for X, vm, mass in test_bar:

        X = X.cuda()
        vm = vm.cuda()
        mass = mass.cuda()

        z1, z2 = nets_(X)

        loss = fit_loss(z1.reshape(-1), vm.reshape(-1)) + \
               fit_loss(z2.reshape(-1), mass.reshape(-1))
        rmae1 = (torch.abs(vm.reshape(-1) - z1.reshape(-1)).sum() / torch.abs(vm).sum()).item()
        rmae2 = (torch.abs(mass.reshape(-1) - z2.reshape(-1)).sum() / torch.abs(mass).sum()).item()
        rmae1_u += torch.abs(vm.reshape(-1) - z1.reshape(-1)).sum().item()
        rmae1_d += torch.abs(vm).sum().item()
        rmae2_u += torch.abs(mass.reshape(-1) - z2.reshape(-1)).sum().item()
        rmae2_d += torch.abs(mass).sum().item()

        total_loss += loss.item()
        vm_rmae += rmae1
        mass_rmae += rmae2

        test_bar.set_description(desc='loss: %.4e,  vm_rmae: %.4e,  mass_rmae: %.4e'
                                      % (loss.item(), rmae1, rmae2))

    print('testing phase: loss: %.4e  vm_rmae: %.4e  mass_rmae: %.4e'
          % (total_loss / len(test_loader),
             rmae1_u / rmae1_d,
             rmae2_u / rmae2_d))