import os
import matplotlib.pyplot as plt
import torch

from nets import *
import numpy as np
from collector import DataLoaderDICT_CLDMP
from tqdm import tqdm
from copy import copy


# 150: cl: 1.0504e-02  cd: 2.6301e-02
# 734: cl: 1.7862e-02  cd: 8.1425e-02

def _init_fn(worker_id):
    np.random.seed(1 + worker_id)


path = 'results-al-semi-734'
# path = 'results-al-semi-150'
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

scales = check['scales']  # torch.ones_like(check['scales'])
scale_cl = scales[0].item()
scale_cd = scales[1].item()

for player in players:
    check_i = torch.load('%s/net_best_%s.tar' % (path, player), map_location=torch.device('cpu'))
    # check_i = torch.load('%s/net_last.tar' % (path, ))
    net_i = ResNet1d18CLDMP()
    net_i.load_state_dict(check_i[player])
    net_i.cpu()
    net_i.eval()
    nets.append(copy(net_i))

MA = int(path.split('-')[-1])
if MA == 150:
    BASE = 101
    ALFAMAX = 10.
else:
    BASE = 101
    ALFAMAX = 3.

data012 = np.load('../config/remake_data_v2_ma%d.npy' % MA, allow_pickle=True).item()

test_ind = np.load('test_ind_ma%d_v2.npy' % MA)
data012['res'] = data012['res'].reshape((-1, BASE, 5))
for k in data012.keys():
    data012[k] = data012[k][test_ind]
data012['res'] = data012['res'].reshape((-1, 5))
data012['res'][:, 0] = data012['res'][:, 0]  # / ALFAMAX
# data012['res'][:, 1] = data012['res'][:, 1] / scale_cl
# data012['res'][:, 2] = data012['res'][:, 2] / scale_cd

test_set = DataLoaderDICT_CLDMP(data012, BASE, 0)
test_set.scale = scales
test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=4,
                                          batch_size=1000, shuffle=False, drop_last=False,
                                          pin_memory=False, worker_init_fn=_init_fn)

fit_loss = nn.L1Loss()


def nets_(X, paras):
    z1 = 0
    z2 = 0
    for i in range(N_nets):
        z12_ = nets[i](X, paras)
        z1 += z12_[:, 0]
        z2 += z12_[:, 1]
    return z1 / N_nets, z2 / N_nets


with torch.no_grad():
    # check = torch.load('%s/net_best.tar' % self.opt['checkpoints'])

    test_bar = tqdm(test_loader)
    total_loss = 0

    rmae1_u = 0
    rmae2_u = 0
    rmae1_d = 0
    rmae2_d = 0

    for X, alfa, cl, cd, cm, cp in test_bar:

        solved = cd >= 0

        z1, z2 = nets_(X, alfa)
        # print(cl, z1)

        loss = fit_loss(z1.reshape(-1), cl.reshape(-1)) + \
               fit_loss(z2.reshape(-1), cd.reshape(-1))
        rmae1 = ((torch.abs(cl.reshape(-1) - z1.reshape(-1))*solved).sum() / torch.abs(cl*solved).sum()).item()
        rmae2 = ((torch.abs(cd.reshape(-1) - z2.reshape(-1))*solved).sum() / torch.abs(cd*solved).sum()).item()

        rmae1_u += torch.abs((cl.reshape(-1) - z1.reshape(-1))*solved).sum().item()
        rmae1_d += torch.abs(cl*solved).sum().item()
        rmae2_u += torch.abs((cd.reshape(-1) - z2.reshape(-1))*solved).sum().item()
        rmae2_d += torch.abs(cd*solved).sum().item()

        total_loss += loss.item()

        test_bar.set_description(desc='loss: %.4e,  cl: %.4e,  cd: %.4e'
                                      % (loss.item(), rmae1, rmae2))

    print('testing phase: loss: %.4e  cl: %.4e  cd: %.4e'
          % (total_loss / len(test_loader),
             rmae1_u / rmae1_d,
             rmae2_u / rmae2_d))
