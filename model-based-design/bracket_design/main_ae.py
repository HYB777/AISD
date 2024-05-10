from model_ae import *
from data_loader import DataLoaderAE
from train import TrainerAE
import torch
import numpy as np
import argparse
import os
import random

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


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='the dir of data set')
parser.add_argument('--batch_size', type=int, default=4096, help='the batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='the learning rate')
parser.add_argument('--nEpoch', type=int, default=10000, help='the number of epochs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--checkpoints', type=str, default='./results_ae', help='folder to output model checkpoints')
parser.add_argument('--checkpoints_weight', type=str, default='', help='model weight')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--embed-dim', type=int, default=16, help='number of data loading workers')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        opt.cuda = True
    opts = {k: v for k, v in opt._get_kwargs()}
    opts['checkpoints'] += 'AE_ID=%d' % (opts['embed_dim'])
    opts['checkpoints'] = 'shape-anomaly-detection/' + opts['checkpoints']
    print(opts)
    try:
        os.makedirs(opts['checkpoints'])
    except OSError:
        pass

    param_train = np.load('config/cpt_train.npy').transpose((0, 2, 1))
    param_val = np.load('config/cpt_val.npy').transpose((0, 2, 1))
    param_test = np.load('config/cpt_test.npy').transpose((0, 2, 1))

    area_train = np.load('config/area_train.npy')
    area_val = np.load('config/area_val.npy')
    area_test = np.load('config/area_test.npy')
    ind_train = area_train > 0
    ind_val = area_val > 0
    ind_test = area_test > 0

    param_train = param_train[ind_train]
    param_val = param_val[ind_val]
    param_test = param_test[ind_test]

    minn = param_train.min((0, 2))
    maxx = param_train.max((0, 2))
    mu = (maxx + minn) / 2
    sig = (maxx - minn) / 2

    param_train = (param_train - mu.reshape(1, 2, 1)) / sig.reshape(1, 2, 1)
    param_val = (param_val - mu.reshape(1, 2, 1)) / sig.reshape(1, 2, 1)
    param_test = (param_test - mu.reshape(1, 2, 1)) / sig.reshape(1, 2, 1)

    opts['mu'] = mu.reshape(1, 2, 1)
    opts['sig'] = sig.reshape(1, 2, 1)

    train_set = DataLoaderAE(param_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=int(opt.workers),
                                               batch_size=opts['batch_size'], shuffle=True, drop_last=True,
                                               worker_init_fn=_init_fn)

    val_set = DataLoaderAE(param_val)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=int(opt.workers),
                                             batch_size=431, shuffle=False, drop_last=False,
                                             worker_init_fn=_init_fn)

    test_set = DataLoaderAE(param_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=int(opt.workers),
                                              batch_size=1, shuffle=False, drop_last=False,
                                              worker_init_fn=_init_fn)

    params_dict = [
        [34, 52, 58, 58],
        [30, 54, 58, 58],
        [36, 50, 58, 58],
        [34, 40, 64, 64],
        [30, 46, 60, 64],
        [30, 48, 58, 64],
        [32, 44, 60, 64],
        [30, 46, 60, 62],
        [30, 48, 58, 62],
        [34, 44, 58, 64],
        [30, 54, 56, 56],
        [32, 48, 56, 62],
        [32, 46, 56, 64],
        [32, 50, 56, 58],
        [32, 36, 64, 64],
        [34, 42, 60, 60],
        [30, 52, 52, 62],
        [30, 50, 52, 64],
        [44, 46, 52, 58],
        [32, 38, 60, 64],
        [30, 44, 58, 60],
        [30, 46, 56, 60],
        [30, 48, 54, 60],
        [32, 32, 64, 64],
    ]

    aecoder = AEcoder(embed_dim=opts['embed_dim'],
                      ndfs=params_dict[opts['embed_dim']-1],
                      ngfs=params_dict[opts['embed_dim']-1])
    trainer = TrainerAE(aecoder, train_loader, val_loader, test_loader, opts)
    trainer.run()

    # errs = []
    #
    # for e in range(24, 25, 2):
    #     aecoder = AEcoder(
    #         ndfs=params_dict[e - 1],
    #         ngfs=params_dict[e - 1],
    #         embed_dim=e
    #     )
    #     print(aecoder)
    #     # input(sum([p.numel() for p in aecoder.parameters()]))
    #     # opts['checkpoints'] = 'learning-results-ae/resultsAE_ID=%d_wgan' % e
    #     opts['checkpoints'] = 'learning-results-ae/results_aeAE_ID=%d' % e
    #     trainer = TrainerAE(aecoder, train_loader, val_loader, test_loader, opts)
    #     err, _ = trainer.test()
    #     errs.append(err)
    # plt.plot(list(range(2, 25, 2)), errs, 'o-')
    # plt.xlabel('dimension of latent space')
    # plt.ylabel('reconstruction error')
    # # plt.scatter(5, errs[4], c='r', marker='o')
    # plt.xticks(list(range(2, 25, 2)))
    # plt.show()
