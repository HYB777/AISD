import matplotlib.pyplot as plt

from nets.ae_nets import *
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
parser.add_argument('--batch_size', type=int, default=256, help='the batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='the learning rate')
parser.add_argument('--nEpoch', type=int, default=10000, help='the number of epochs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--checkpoints', type=str, default='results', help='folder to output model checkpoints')
parser.add_argument('--checkpoints_weight', type=str, default='', help='model weight')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--embed-dim', type=int, default=16, help='number of data loading workers')


if __name__ == '__main__':
    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        opt.cuda = True
    opts = {k: v for k, v in opt._get_kwargs()}
    opts['checkpoints'] += 'AE_ID=%d_wgan' % (opts['embed_dim'])
    opts['checkpoints'] = 'shape-anomaly-detection' + opts['checkpoints']
    print(opts)
    try:
        os.makedirs(opts['checkpoints'])
    except OSError:
        pass

    train_size = 40000
    val_size = train_size + 3000
    test_size = val_size + 0
    
    data_x = np.load('config/data_extend_wgan.npy').astype(np.float32)

    data_x = data_x.reshape(-1, 2, 32)
    inds = np.arange(data_x.shape[0])
    np.random.shuffle(inds)
    min_ = data_x[inds[:train_size]].min((0, 2))
    max_ = data_x[inds[:train_size]].max((0, 2))
    mu = (min_ + max_) / 2.
    sig = (max_ - min_) / 2.
    data_x = (data_x - mu.reshape(1, 2, 1)) / sig.reshape(1, 2, 1)

    mu = torch.from_numpy(mu.reshape(1, 2, 1))
    sig = torch.from_numpy(sig.reshape(1, 2, 1))

    train_x = torch.from_numpy(data_x[inds[:train_size]])
    val_x = torch.from_numpy(data_x[inds[train_size:val_size]])
    test_x = torch.from_numpy(data_x[inds[val_size:]])

    train_set = DataLoaderAE(train_x)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=int(opt.workers),
                                               batch_size=opts['batch_size'], shuffle=True, drop_last=True,
                                               worker_init_fn=_init_fn)

    val_set = DataLoaderAE(val_x)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=int(opt.workers),
                                             batch_size=1, shuffle=False, drop_last=True,
                                             worker_init_fn=_init_fn)

    test_set = DataLoaderAE(test_x)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=int(opt.workers),
                                              batch_size=1, shuffle=False, drop_last=True,
                                              worker_init_fn=_init_fn)

    params_dict = [
        [7, 7, 8, 10], 
        [4, 6, 10, 10],
        [4, 8, 8, 10],
        [4, 5, 10, 10],
        [4, 7, 8, 10],
        [5, 7, 7, 10],
        [4, 8, 8, 8],
        [5, 6, 7, 10],
        [3, 4, 9, 10],
        [5, 6, 7, 9],
        [6, 6, 7, 8],
        [3, 5, 7, 10],  
        [4, 5, 7, 9],
        [4, 6, 7, 8],
        [4, 4, 7, 9],
        [4, 4, 8, 8],
    ]

    aecoder = AEcoder(
        ndfs=params_dict[opts['embed_dim']-1],
        ngfs=params_dict[opts['embed_dim']-1],
        embed_dim=opts['embed_dim']
    )
    # input(sum([p.numel() for p in aecoder.parameters()]))
    trainer = TrainerAE(aecoder, train_loader, val_loader, test_loader, opts, mu, sig)
    trainer.run()

    #
    # errs = []
    # for e in range(1, 17):
    #     aecoder = AEcoder(
    #         ndfs=params_dict[e - 1],
    #         ngfs=params_dict[e - 1],
    #         embed_dim=e
    #     )
    #     # input(sum([p.numel() for p in aecoder.parameters()]))
    #     opts['checkpoints'] = 'shape-anomaly-detection/resultsAE_ID=%d_wgan' % e
    #     trainer = TrainerAE(aecoder, train_loader, val_loader, test_loader, opts, mu, sig)
    #     err, _ = trainer.test()
    #     errs.append(err)
    # plt.plot(list(range(1, 17)), errs, 'o-')
    # plt.xlabel('dimension of latent space')
    # plt.ylabel('reconstruction error')
    # # plt.scatter(5, errs[4], c='r', marker='o')
    # plt.xticks(list(range(1, 17)))
    # plt.show()
