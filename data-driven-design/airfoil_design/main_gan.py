from nets.gan_nets import *
from data_loader import DataLoaderGAN
from train import TrainerGAN
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
parser.add_argument('--embed_dim', type=int, default=16, help='the batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='the learning rate')
parser.add_argument('--lambda_gp', type=float, default=10, help='the learning rate')
parser.add_argument('--regular', type=float, default=0.0, help='the learning rate')
parser.add_argument('--nEpoch', type=int, default=1000, help='the number of epochs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--wass', action='store_true', help='enables wasserstein')
parser.add_argument('--checkpoints', type=str, default='resultsWGAN', help='folder to output model checkpoints')
parser.add_argument('--checkpoints_weight', type=str, default='', help='model weight')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--ndf', type=int, default=64, help='number of data loading workers')
parser.add_argument('--ngf', type=int, default=64, help='number of data loading workers')


if __name__ == '__main__':
    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        opt.cuda = True
    opts = {k: v for k, v in opt._get_kwargs()}
    opts['checkpoints'] += 'GAN_ngf*ndf=%d*%d_wass_%s' % (opts['ngf'], opts['ndf'], opts['wass'])
    opts['checkpoints'] = 'learning_results/shape-sampling' + opts['checkpoints']
    print(opts)
    try:
        os.makedirs(opts['checkpoints'])
    except OSError:
        pass

    train_size = 1200
    val_size = train_size + 200
    test_size = val_size + 0
    data_x = torch.from_numpy(np.load('data_bs/bs_datas.npy').astype(np.float32))
    # data_x = data_x[torch.arange(data_x.size(0)) != 790]
    inds = np.arange(data_x.shape[0])
    np.random.shuffle(inds)
    min_ = data_x.min()
    max_ = data_x.max()
    mu = (min_ + max_) / 2.
    sig = (max_ - min_) / 2.
    mu = torch.zeros_like(mu)
    sig = torch.ones_like(sig)
    data_x = (data_x - mu) / sig

    train_x = data_x[inds[:train_size]]
    val_x = data_x[inds[train_size:val_size]]
    test_x = data_x[inds[val_size:]]

    train_set = DataLoaderGAN(train_x)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=int(opt.workers),
                                               batch_size=opts['batch_size'], shuffle=True, drop_last=True,
                                               worker_init_fn=_init_fn)

    val_set = DataLoaderGAN(val_x)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=int(opt.workers),
                                             batch_size=100, shuffle=False, drop_last=True,
                                             worker_init_fn=_init_fn)

    test_set = DataLoaderGAN(test_x)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=int(opt.workers),
                                              batch_size=100, shuffle=False, drop_last=True,
                                              worker_init_fn=_init_fn)

    netD = DiscriminatorCONV(opts['ndf'])
    netG = GeneratorCONV(opts['ngf'], embed_dim=opts['embed_dim'])
    trainer = TrainerGAN(netD, netG, train_loader, val_loader, test_loader, opts, mu, sig, max_, min_)
    trainer.run()

