import numpy as np
import torch
import torch.nn as nn
from typing import Any, Sequence, Tuple, Union
import torch.nn.functional as F
from torchvision.models.resnet import *


class DiscriminatorCONV(nn.Module):
    def __init__(self, ndf):
        super(DiscriminatorCONV, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.AvgPool1d(2)
        self.conv5 = nn.Conv1d(ndf * 8, 1, 2, 1, 0, bias=False)

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

        return x.reshape(b, -1)


class GeneratorCONV(nn.Module):
    def __init__(self, ngf, embed_dim=16):
        super(GeneratorCONV, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, ngf * 8, 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf * 2, ngf * 1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf, 2, 3, 1, 1, bias=False),
            # nn.Tanh(),
        )

    def forward(self, x):

        x = self.conv1(x.unsqueeze(-1))

        x = self.upsample(x)
        x = self.conv2(x)

        x = self.upsample(x)
        x = self.conv3(x)

        x = self.upsample(x)
        x = self.conv4(x)

        x = self.upsample(x)
        x = self.conv5(x)
        return x

