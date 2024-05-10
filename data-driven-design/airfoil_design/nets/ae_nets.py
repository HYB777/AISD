import numpy as np
import torch
import torch.nn as nn
from typing import Any, Sequence, Tuple, Union
import torch.nn.functional as F
from torchvision.models.resnet import *

USE_BIAS = False


class AEcoder(nn.Module):
    def __init__(self, ndfs, ngfs, embed_dim=16):
        super(AEcoder, self).__init__()
        self.encoder = Encoder(ndfs, embed_dim)
        self.decoder = Decoder(ngfs, embed_dim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class Encoder(nn.Module):
    def __init__(self, ndfs, embed_dim=16):
        super(Encoder, self).__init__()
        ndf1, ndf2, ndf3, ndf4 = ndfs
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, ndf1, 3, 1, 1, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(ndf1, ndf2, 3, 1, 1, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(ndf2, ndf3, 3, 1, 1, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(ndf3, ndf4, 3, 1, 1, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.AvgPool1d(2)
        self.conv5 = nn.Conv1d(ndf4, embed_dim, 2, 1, 0, bias=USE_BIAS)

    def forward(self, x):
        b = x.shape[0]
        x = self.conv1(x)
        x = self.downsample(x)

        x = self.conv2(x)
        x = self.downsample(x)

        x = self.conv3(x)
        x = self.downsample(x)

        x = self.conv4(x)
        x = self.downsample(x)

        x = self.conv5(x)

        return x


class Decoder(nn.Module):
    def __init__(self, ngfs, embed_dim):
        super(Decoder, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        ngf1, ngf2, ngf3, ngf4 = ngfs
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, ngf4, 2, 1, 0, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            # nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf4, ngf3, 3, 1, 1, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf3, ngf2, 3, 1, 1, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            # nn.Conv1d(ngf * 1, ngf * 1, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf2, ngf1, 3, 1, 1, bias=USE_BIAS),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            # nn.Conv1d(ngf, ngf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf1, 2, 3, 1, 1, bias=USE_BIAS),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.upsample(x)
        x = self.conv2(x)

        x = self.upsample(x)
        x = self.conv3(x)

        x = self.upsample(x)
        x = self.conv4(x)

        x = self.upsample(x)
        x = self.conv5(x)

        return x


if __name__ == '__main__':
    from copy import deepcopy
    from tqdm import tqdm
    ndfs = [4, 4, 8, 8]
    ngfs = [4, 4, 8, 8]
    embed = 16
    ae = AEcoder(ndfs, ngfs, embed_dim=embed)
    n_params = sum([p.numel() for p in ae.parameters()])
    print(n_params)
    hyper_dict = {}
    for e in tqdm(range(1, 17)):
        err = np.inf
        prev_n = np.inf
        hyperparam_e = []
        for a in range(2, 11):
            for b in range(a, 11):
                for c in range(b, 11):
                    for d in range(c, 11):
                        ndfs = [a, b, c, d]
                        ngfs = [a, b, c, d]
                        ae = AEcoder(ndfs, ngfs, embed_dim=e)
                        cur_n_params = sum([p.numel() for p in ae.parameters()])
                        # if abs(n_params - cur_n_params) <= err:
                        #     if abs(n_params - cur_n_params) == err:
                        #         if cur_n_params <= n_params:
                        #             hyperparam_e.append(deepcopy(ndfs))
                        #     else:
                        #         err = abs(n_params - cur_n_params)
                        #         prev_n = cur_n_params
                        #         hyperparam_e = [deepcopy(ndfs)]
                        # if 0 <= (n_params - cur_n_params) <= err:
                        #     if (n_params - cur_n_params) == err:
                        #         hyperparam_e.append(deepcopy(ndfs))
                        #     else:
                        #         err = (n_params - cur_n_params)
                        #         hyperparam_e = [deepcopy(ndfs)]
                        if 0 <= abs(n_params - cur_n_params) <= 10 and cur_n_params <= n_params:
                            hyperparam_e.append(deepcopy(ndfs))

        hyper_dict[e] = deepcopy(hyperparam_e)

    for k, v in hyper_dict.items():
        print(k, [(vi, sum([p.numel() for p in AEcoder(vi, vi, k).parameters()])) for vi in v])



"""

1 [[5, 7, 9, 10]] 1228
2 [[3, 8, 9, 10], [4, 6, 10, 10], [6, 6, 9, 10]] 1232
3 [[5, 5, 10, 10]] 1230
4 [[4, 5, 10, 10]] 1228
5 [[1, 8, 9, 10], [4, 7, 8, 10]] 1232
6 [[2, 5, 10, 10], [2, 8, 8, 10], [4, 4, 10, 10], [5, 7, 7, 10], [6, 6, 8, 9]] 1224
7 [[4, 8, 8, 8], [7, 7, 7, 8]] 1232
8 [[1, 6, 9, 10], [2, 4, 10, 10], [3, 6, 8, 10], [5, 6, 7, 10]] 1232
9 [[1, 7, 8, 10], [2, 6, 9, 9], [3, 3, 10, 10]] 1230
10 [[5, 6, 8, 8]] 1232
11 [[1, 4, 9, 10], [3, 3, 9, 10]] 1232
12 [[1, 5, 9, 9], [3, 5, 8, 9], [4, 7, 7, 8], [5, 5, 6, 10], [5, 5, 7, 9]] 1230
13 [[3, 6, 8, 8]] 1232
14 [[1, 2, 9, 10], [1, 5, 7, 10], [2, 6, 6, 10]] 1232
15 [[3, 5, 8, 8], [4, 4, 7, 9]] 1230
16 [[1, 6, 8, 8], [4, 4, 8, 8]] 1232

Process finished with exit code 0


Process finished with exit code 0


"""