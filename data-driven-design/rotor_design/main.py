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
        return self.trainers.sample_in_uncertainty(N, subset)

    def save_state_dict(self):
        torch.save(self.trainers.state_dict(), 'results_al/checkpoints_last.tar')

    def run(self, init_num=1000, add_num=500, resume_ind=None, run_steps=None):

        if resume_ind is None:
            self.trainers.collector.random_add_from_pool(init_num, split_ratio=10)
            try:
                init_check = torch.load('init-%dW.tar' % (init_num//1_0000))
                self.trainers.load_state_dict(init_check, False)
            except FileNotFoundError as e:
                input(e)
                init_check = self.trainers.state_dict()
                torch.save(init_check, 'init-%dW.tar' % (init_num//1_0000))
        else:
            try:
                check = torch.load('checkpoints_last.tar')
                self.trainers.load_state_dict(check)
            except Exception as e:
                print(e)

        B = 0
        while not self.trainers.collector.terminated():
            print('#################################collect size: %d#################################' % self.collector_size())
            self.run_trainers(300, 0)
            # self.run_trainers(2, 0)
            with torch.no_grad():
                r_ind, r_uncertainty = self.compute_uncertainty(N=1)
            # input(r_uncertainty.shape)
            add_num_nb = add_num // len(r_ind.keys())
            r_uncertainty_mean = 0
            for nb in r_ind.keys():
                r_uncertainty_nb = r_uncertainty[nb]
                r_ind_nb = r_ind[nb][np.argsort(-r_uncertainty_nb)]
                r_uncertainty_mean += r_uncertainty_nb.mean()
                self.trainers.collector.add_from_pool(r_ind_nb[:add_num_nb], update_scales_flag=False, split_ratio=10)

            self.uncertainty.append(r_uncertainty_mean / len(r_ind.keys()))
            self.save_state_dict()
            np.savetxt('uncertainty.txt', np.array(self.uncertainty))
            B += 1

        self.trainers.collector.save_dataset_ind('inds.npy')

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
parser.add_argument('--checkpoints', type=str, default='./results_al', help='folder to output model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')


if __name__ == '__main__':
    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        opt.cuda = True
    opts = {k: v for k, v in opt._get_kwargs()}
    opts['_init_fn'] = _init_fn
    print(opts)
    try:
        os.makedirs(opts['checkpoints'])
    except OSError:
        pass

    data_dict = {}

    pitches = np.load('rotors_dataset/pitches.npy')
    airfoils = np.load('airfoils/conts_wgan.npy')
    perms = np.load('rotors_dataset/rotor_keys.npy')
    res = np.load('rotors_dataset/rotor_values.npy')

    train_val_ind = np.load('train_val_ind.npy')
    perms = perms[train_val_ind[:len(train_val_ind)//2]]
    res = res[train_val_ind[:len(train_val_ind)//2]]

    mu = pitches.mean(0)
    sig = pitches.std(0)
    pitches = (pitches - mu) / sig

    MAX_NUM = 50_0000

    SEMI_SUPERVISED = True
    data_set = DataLoaderDICT_CTQ(
        {
            'airfoil': airfoils,
            'pitches': pitches,
            'perms': perms,
            'res': res,
        }
    )
    data_collector = Collector(data_set, max_num=MAX_NUM)
    trainer = TrainerFitLL(data_collector, opts, mu, sig, semi_supervised=SEMI_SUPERVISED)

    act = ActivatorSemiSupervisor(trainers=trainer, opt=opt)
    # act.run(init_num=1000, add_num=500)  # 3540800
    act.run(init_num=10_0000, add_num=10000)
