import numpy as np
from train import TrainerAE
from models_ae import *
from data_loader import *
import argparse
import os
import random
from tqdm import tqdm

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
parser.add_argument('--batch_size', type=int, default=1024, help='the batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument('--regular', type=float, default=0.0, help='the learning rate')
parser.add_argument('--nEpoch', type=int, default=1000, help='the number of epochs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--checkpoints', type=str, default='results', help='folder to output model checkpoints')
parser.add_argument('--checkpoints_weight', type=str, default='', help='model weight')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--embed_dim', type=int, default=16, help='number of data loading workers')

if __name__ == '__main__':
    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        opt.cuda = True

    opts = {k: v for k, v in opt._get_kwargs()}
    opts['checkpoints'] = 'shape-anomaly-detection/results_aeID=%d' % opts['embed_dim']
    print(opts)
    try:
        os.makedirs(opts['checkpoints'])
    except OSError:
        pass

    try:
        data_train = np.load('%s/data_train.npy' % opts['data'], allow_pickle=True).item()
        data_val = np.load('%s/data_val.npy' % opts['data'], allow_pickle=True).item()
        data_test = np.load('%s/data_test.npy' % opts['data'], allow_pickle=True).item()
    except FileNotFoundError:
        mat_str_list = [opts['data'] + '/' + it for it in os.listdir(opts['data']) if it.endswith('.mat')]
        params, labels = mat2npy(mat_str_list)

        total_size = params.shape[0]
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size) + train_size
        ind = np.random.permutation(total_size)

        data_train = {'params': params[ind[:train_size]], 'labels': labels[ind[:train_size]]}
        data_val = {'params': params[ind[train_size:val_size]], 'labels': labels[ind[train_size:val_size]]}
        data_test = {'params': params[ind[val_size:]], 'labels': labels[ind[val_size:]]}

        np.save('%s/data_train.npy' % opts['data'], data_train)
        np.save('%s/data_val.npy' % opts['data'], data_val)
        np.save('%s/data_test.npy' % opts['data'], data_test)

    mu = data_train['params'].mean(0)
    sig = data_train['params'].std(0)

    data_train['params'] = (data_train['params'] - mu) / sig
    data_val['params'] = (data_val['params'] - mu) / sig
    data_test['params'] = (data_test['params'] - mu) / sig

    train_set = DataLoaderAE(data_train['params'])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=int(opt.workers),
                                               batch_size=opts['batch_size'], shuffle=True, drop_last=True,
                                               worker_init_fn=_init_fn)

    val_set = DataLoaderAE(data_val['params'])
    val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=int(opt.workers),
                                             batch_size=1000, shuffle=False, drop_last=True,
                                             worker_init_fn=_init_fn)

    test_set = DataLoaderAE(data_test['params'])
    test_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=int(opt.workers),
                                              batch_size=1, shuffle=False, drop_last=False,
                                              worker_init_fn=_init_fn)

    params_dict = [[73, 73, 73, 73] for _ in range(1, 73)]

    net = AERes(embed_dim=opts['embed_dim'],
                ndfs=params_dict[opts['embed_dim']-1],
                ngfs=params_dict[opts['embed_dim']-1])
    trainer = TrainerAE(net, train_loader, val_loader, test_loader, opts, mu, sig)
    trainer.run()
    # errs = []
    # import matplotlib.pyplot as plt
    # range_ = range(72, 73, 4)
    # res = sorted([int(ei.split('=')[-1]) for ei in os.listdir('learning_results-ae')])
    # print(res)
    # for ei in res:
    #     opts['checkpoints'] = 'learning_results-ae/results_aeID=%d' % ei
    #     net = AERes(embed_dim=ei,
    #                 ndfs=[96, 96, 96, 96],
    #                 ngfs=[96, 96, 96, 96])
    #     # print(net)
    #
    #     trainer = TrainerAE(net, train_loader, val_loader, test_loader, opts, mu, sig)
    #     err, _ = trainer.test()
    #     errs.append(err)
    #     # input()
    # plt.plot(list(res), errs, 'o-')
    # plt.xlabel('dimension of latent space')
    # plt.ylabel('reconstruction error')
    # # plt.scatter(5, errs[4], c='r', marker='o')
    # plt.xticks(list(res))
    # plt.axis('equal')
    # np.save('x_res.npy', list(res))
    # np.save('errs.npy', errs)
    # plt.show()
