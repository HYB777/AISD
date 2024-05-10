import numpy as np
import torch
import torch.nn as nn
from typing import Any, Sequence, Tuple, Union
import torch.nn.functional as F
import torchvision
torchvision.models.resnet18()


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1, bias=True):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm1d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm1d(n_out)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet1d18Fit(nn.Module):
    def __init__(self):
        super(ResNet1d18Fit, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv1d(2*3*3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.forw1 = nn.Sequential(
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
        )
        self.forw2 = nn.Sequential(
            PostRes(64, 128, stride=2, bias=False),
            PostRes(128, 128, bias=False),
        )
        self.forw3 = nn.Sequential(
            PostRes(128, 256, stride=2, bias=False),
            PostRes(256, 256, bias=False),
        )
        self.forw4 = nn.Sequential(
            PostRes(256, 512, stride=2, bias=False),
            PostRes(512, 512, bias=False),
        )
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.vm_net = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1, bias=True)
        )
        self.mass_net = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1, bias=True)
        )

    def ema_update(self, other, eta):
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            self_param.data.mul_(eta).add_(other_param.data, alpha=1 - eta)

    def forward(self, x):
        # (N, 1, 64, 64)
        b = x.shape[0]
        out = self.preBlock(x.reshape(b, 3*3*2, -1))                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        # out5 = self.avg_pool(out4)

        out5 = torch.flatten(out4, 1)

        vm = self.vm_net(out5)
        mass = self.mass_net(out5)

        out = torch.cat([vm.view(-1, 1), mass.view(-1, 1)], dim=1)

        return torch.abs(out)

