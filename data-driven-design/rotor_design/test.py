import os
import matplotlib.pyplot as plt
import torch

from nets import *
import numpy as np
from collector import DataLoaderDICT_CTQ, sub_dataset
from tqdm import tqdm
from copy import copy


def _init_fn(worker_id):
    np.random.seed(1 + worker_id)


# ct: 2.8978e-02  cq: 2.8511e-02
path = 'results_al'
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
print(scales)
scale_cl = scales[0].item()
scale_cd = scales[1].item()

for player in players:
    check_i = torch.load('%s/net_best_%s.tar' % (path, player), map_location=torch.device('cpu'))
    # check_i = torch.load('%s/net_last.tar' % (path, ))
    net_i = ResNet1d18_Rotor()
    net_i.load_state_dict(check_i[player])
    net_i.cpu()
    net_i.eval()
    nets.append(copy(net_i))

data_dict = {}

pitches = np.load('rotors_dataset/pitches.npy')
airfoils = np.load('airfoils/conts_wgan.npy')
perms = np.load('rotors_dataset/rotor_keys.npy')
res = np.load('rotors_dataset/rotor_values.npy')

train_val_ind = np.load('test_ind.npy')
perms = perms[train_val_ind]
res = res[train_val_ind]

mu = pitches.mean(0)
sig = pitches.std(0)
pitches = (pitches - mu) / sig

data_set = DataLoaderDICT_CTQ(
    {
        'airfoil': airfoils,
        'pitches': pitches,
        'perms': perms,
        'res': res,
    }
)
data_set.scale = scales

# val_ind = torch.load('%s/checkpoints_last.tar' % path)['collector']['val_ind']
# data_set_ = sub_dataset(data_set, val_ind)

test_loader = torch.utils.data.DataLoader(dataset=data_set, num_workers=4,
                                          batch_size=1000, shuffle=False, drop_last=False,
                                          pin_memory=False, worker_init_fn=_init_fn)

fit_loss = nn.L1Loss()


def nets_(X, n_blade, pitches):
    z1 = 0
    z2 = 0
    for i in range(N_nets):
        z12_ = nets[i](X, n_blade, pitches)
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

    for X, n_blade, pitches, ct, cq in test_bar:

        # assert cd >= 0

        z1, z2 = nets_(X, n_blade, pitches)
        # print(cl, z1)

        loss = fit_loss(z1.reshape(-1), ct.reshape(-1)) + \
               fit_loss(z2.reshape(-1), cq.reshape(-1))
        rmae1 = ((torch.abs(ct.reshape(-1) - z1.reshape(-1))).sum() / torch.abs(ct).sum()).item()
        rmae2 = ((torch.abs(cq.reshape(-1) - z2.reshape(-1))).sum() / torch.abs(cq).sum()).item()

        rmae1_u += torch.abs((ct.reshape(-1) - z1.reshape(-1))).sum().item()
        rmae1_d += torch.abs(ct).sum().item()
        rmae2_u += torch.abs((cq.reshape(-1) - z2.reshape(-1))).sum().item()
        rmae2_d += torch.abs(cq).sum().item()

        total_loss += loss.item()

        test_bar.set_description(desc='loss: %.4e,  ct: %.4e,  cq: %.4e'
                                      % (loss.item(), rmae1, rmae2))

    print('testing phase: loss: %.4e  ct: %.4e  cq: %.4e'
          % (total_loss / len(test_loader),
             rmae1_u / rmae1_d,
             rmae2_u / rmae2_d))
