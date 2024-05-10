import torch
from torch import nn
import torch.nn.functional as F


class AEcoder(nn.Module):
    def __init__(self, embed_dim, ndfs, ngfs):
        super(AEcoder, self).__init__()
        self.encoder = Encoder(ndfs, embed_dim)
        self.decoder = Decoder(ngfs, embed_dim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class Encoder(nn.Module):
    def __init__(self, ndfs, embed_dim):
        super(Encoder, self).__init__()
        ndf1, ndf2, ndf3, ndf4 = ndfs
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, ndf1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(ndf1, ndf2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(ndf2, ndf3, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(ndf3, ndf4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.AvgPool1d(2)
        self.conv5 = nn.Conv1d(ndf4, embed_dim, 2, 1, 0, bias=False)

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
        ngf1, ngf2, ngf3, ngf4 = ngfs
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, ngf4, 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            # nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf4, ngf3, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf3, ngf2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            # nn.Conv1d(ngf * 1, ngf * 1, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf2, ngf1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            # nn.Conv1d(ngf, ngf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf1, 2, 3, 1, 1, bias=False),
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
    import numpy as np
    ndfs = [32, 32, 64, 64]
    ngfs = [32, 32, 64, 64]
    embed = 16
    ae = AEcoder(embed, ndfs, ngfs)
    n_params = sum([p.numel() for p in ae.parameters()])
    print(n_params)
    hyper_dict = {}
    for e in tqdm(range(1, 17)):
        err = np.inf
        prev_n = np.inf
        hyperparam_e = []
        for a in range(30, 65, 2):
            for b in range(a, 65, 2):
                for c in range(b, 65, 2):
                    for d in range(c, 65, 2):
                        ndfs = [a, b, c, d]
                        ngfs = [a, b, c, d]
                        ae = AEcoder(e, ndfs, ngfs)
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
                        if 0 <= abs(n_params - cur_n_params) <= 32 and cur_n_params <= n_params:
                            hyperparam_e.append(deepcopy(ndfs))
                            # print(hyperparam_e)

        hyper_dict[e] = deepcopy(hyperparam_e)

    for k, v in hyper_dict.items():
        print(k, [(vi, sum([p.numel() for p in AEcoder(k, vi, vi).parameters()])) for vi in v])