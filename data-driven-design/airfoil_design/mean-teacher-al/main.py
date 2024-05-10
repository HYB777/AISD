import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from collector import *
from train_al import TrainerFitLL
from nets import *
import argparse
import os
import random
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


N_PARAMS = 14
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def _init_fn(worker_id):
    np.random.seed(seed + worker_id)


class ActivatorSemiSupervisor:
    def __init__(self, trainers, opt):
        self.trainers = trainers
        self.opt = opt
        self.uncertainty = []

    def collector_size(self):
        return len(self.trainers.collector.dataset_ind)

    def run_trainers(self, n_epoch, convergence_epoch=100):
        self.trainers.run(n_epoch, convergence_epoch)

    def compute_uncertainty(self, N=100, subset=-1):
        return self.trainers.sample_in_uncertainty(N)

    def save_state_dict(self):
        torch.save(self.trainers.state_dict(), '%s/checkpoints_last.tar' % self.opt['checkpoints'])

    def run(self, init_num=1000, add_num=500, resume_ind=None, run_steps=None):

        if resume_ind is None:
            self.trainers.collector.random_add_from_pool(init_num, split_ratio=10)
            try:
                init_check = torch.load('init-%dW-ma%d.tar' % (init_num//1_0000, self.opt['ma']))
                self.trainers.load_state_dict(init_check, False)
            except FileNotFoundError as e:
                input(e)
                init_check = self.trainers.state_dict()
                torch.save(init_check, 'init-%dW-ma%d.tar' % (init_num//1_0000, self.opt['ma']))
        else:
            try:
                check = torch.load('checkpoints_last.tar')
                self.trainers.load_state_dict(check)
            except Exception as e:
                print(e)

        B = 0
        while not self.trainers.collector.terminated():
            print('#################################collect size: %d#################################' % self.collector_size())
            self.run_trainers(250, 0)
            # self.run_trainers(1, 0)
            with torch.no_grad():
                r_ind, r_uncertainty = self.compute_uncertainty(N=1)
            # input(r_uncertainty.shape)
            self.uncertainty.append(r_uncertainty.mean())
            r_ind = r_ind[np.argsort(-r_uncertainty)]
            filter_ind = self.trainers.collector.filter(r_ind[:add_num])
            self.trainers.collector.add_from_pool(filter_ind, update_scales_flag=False, split_ratio=10)  #...
            self.save_state_dict()
            np.savetxt('uncertainty-ma%d.txt' % self.opt['ma'], np.array(self.uncertainty))
            B += 1
            # self.run_trainers(1, 0)

        self.trainers.collector.save_dataset_ind('inds-ma%d.npy' % self.opt['ma'])

        # global_steps = run_steps if run_steps is not None else 0
        self.run_trainers(500, 100)
        self.save_state_dict()

        # n_epoch = max(max(len(self.collector)//250, 200), global_steps)
        # n_epoch_ = round(n_epoch / 200)
        # for i in range(n_epoch_):
        #     self.run_trainers(200, False)
        #     self.save_state_dict()


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='the dir of data set')
parser.add_argument('--batch_size', type=int, default=10000, help='the batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate')
parser.add_argument('--nEpoch', type=int, default=1000, help='the number of epochs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--bn', action='store_true', help='enables bn')
parser.add_argument('--checkpoints', type=str, default='./results_al', help='folder to output model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--ma', type=int, default=734, help='mach number * 1000')


if __name__ == '__main__':
    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        opt.cuda = True
    opts = {k: v for k, v in opt._get_kwargs()}
    opts['_init_fn'] = _init_fn
    opts['checkpoints'] += '_%dX' % opts['ma']
    
    print(opts)
    try:
        os.makedirs(opts['checkpoints'])
    except OSError:
        pass

    MA = opts['ma']
    if MA == 150:
        BASE = 101
        ALFAMAX = 10.
        eps = 0.1 / 2
    else:
        BASE = 101
        ALFAMAX = 3.
        eps = 0.03 / 2

    # data0 = np.load('../config/remake_data_0_ma%d.npy' % MA, allow_pickle=True).item()
    # data1 = np.load('../config/remake_data_1_ma%d.npy' % MA, allow_pickle=True).item()
    # data2 = np.load('../config/remake_data_2_ma%d.npy' % MA, allow_pickle=True).item()
    # data012 = {}
    # for k in data0.keys():
    #     data012[k] = np.vstack([data0[k], data1[k], data2[k]])

    data012 = np.load('../config/remake_data_v2_ma%d.npy' % MA, allow_pickle=True).item()
    train_val_ind = np.load('train_val_ind_ma%d_v2.npy' % MA)
    data012['res'] = data012['res'].reshape((-1, BASE, 5))
    for k in data012.keys():
        data012[k] = data012[k][train_val_ind]
    data012['res'] = data012['res'].reshape((-1, 5))
    # data012['res'][:, 0] = data012['res'][:, 0] / ALFAMAX

    MAX_NUM = 50_0000

    SEMI_SUPERVISED = True
    data_set = DataLoaderDICT_CLDMP(copy(data012), BASE, eps)
    data_collector = Collector(data_set, max_num=MAX_NUM)
    trainer = TrainerFitLL(data_collector, opts, semi_supervised=SEMI_SUPERVISED)

    act = ActivatorSemiSupervisor(trainers=trainer, opt=opts)
    # act.run(init_num=1000, add_num=500)  # 3540800
    act.run(init_num=10_0000, add_num=10000)

    # act.run(init_num=5_0000, add_num=1000, run_steps=10088)
    # 16352

    # 3789000
