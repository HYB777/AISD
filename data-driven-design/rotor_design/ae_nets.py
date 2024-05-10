import numpy as np
import torch
import torch.nn as nn
from typing import Any, Sequence, Tuple, Union
import torch.nn.functional as F
from torchvision.models.resnet import *


class AEcoder(nn.Module):
    def __init__(self, ndf, ngf):
        super(AEcoder, self).__init__()
        self.encoder = Encoder(ndf)
        self.decoder = Decoder(ngf, ndf*4)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class Encoder(nn.Module):
    def __init__(self, ndf):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.AvgPool1d(2)
        self.conv5 = nn.Conv1d(ndf * 2, ndf * 4, 2, 1, 0, bias=False)

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
    def __init__(self, ngf, embed_dim):
        super(Decoder, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, ngf * 2, 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            # nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf * 2, ngf * 1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            # nn.Conv1d(ngf * 1, ngf * 1, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf * 1, ngf * 1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            # nn.Conv1d(ngf, ngf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf, 2, 3, 1, 1, bias=False),
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

