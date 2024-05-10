import numpy as np
import torch
import torch.nn as nn
from typing import Any, Sequence, Tuple, Union
import torch.nn.functional as F
import torchvision
torchvision.models.resnet18()


class ResBlockFC(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act: nn.Module = nn.ReLU(True)):
        super(ResBlockFC, self).__init__()
        self.residual_layers = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            act,
            nn.Linear(out_dim, out_dim, bias=True),
        )
        self.short_cut = nn.Linear(in_dim, out_dim, bias=True) if in_dim != out_dim else nn.Identity()
        self.act = act

    def forward(self, x):
        y = self.residual_layers(x)
        z = self.short_cut(x)
        return self.act(y + z)


class ResBlockConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act: nn.Module = nn.ReLU(True)):
        super(ResBlockConv, self).__init__()
        self.residual_layers = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 3, 1, 1, bias=True),
            act,
            nn.Conv1d(out_dim, out_dim, 3, 1, 1, bias=True),
        )
        self.short_cut = nn.Conv1d(in_dim, out_dim, 1, 1, 0, bias=True) if in_dim != out_dim else nn.Identity()
        self.act = act

    def forward(self, x):
        y = self.residual_layers(x)
        z = self.short_cut(x)
        return self.act(y + z)


class AEcoderResNavie(nn.Module):
    def __init__(self):
        super(AEcoderResNavie, self).__init__()
        self.encoder1 = nn.Sequential(
            ResBlockFC(11, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 16, nn.LeakyReLU(0.2, True)),
        )
        self.decoder1 = nn.Sequential(
            ResBlockFC(16, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 11, nn.LeakyReLU(0.2, True)),
        )

        self.encoder2 = nn.Sequential(
            ResBlockFC(55, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 16, nn.LeakyReLU(0.2, True)),
        )
        self.decoder2 = nn.Sequential(
            ResBlockFC(16, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 55, nn.LeakyReLU(0.2, True)),
        )

        self.encoder3 = nn.Sequential(
            ResBlockFC(8, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 16, nn.LeakyReLU(0.2, True)),
        )
        self.decoder3 = nn.Sequential(
            ResBlockFC(16, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 128, nn.LeakyReLU(0.2, True)),
            ResBlockFC(128, 8, nn.LeakyReLU(0.2, True)),
        )

    def forward(self, x):
        x1 = x[:, :11]
        x2 = x[:, 11:66]
        x3 = x[:, 66:]
        z1 = self.encoder1(x1)
        y1 = self.decoder1(z1)
        z2 = self.encoder2(x2)
        y2 = self.decoder2(z2)
        z3 = self.encoder3(x3)
        y3 = self.decoder3(z3)
        y = torch.cat([y1, y2, y3], dim=1)
        return y


class AEcoderRes(nn.Module):
    def __init__(self):
        super(AEcoderRes, self).__init__()
        self.encoder1 = nn.Sequential(
            ResBlockFC(11, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 16, nn.LeakyReLU(0.2, True)),
        )
        self.decoder1 = nn.Sequential(
            ResBlockFC(16, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 11, nn.LeakyReLU(0.2, True)),
        )

        self.encoder2 = nn.Sequential(
            ResBlockFC(5, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 16, nn.LeakyReLU(0.2, True)),
            ResBlockFC(16, 8, nn.LeakyReLU(0.2, True)),
        )
        self.decoder2 = nn.Sequential(
            ResBlockFC(8, 16, nn.LeakyReLU(0.2, True)),
            ResBlockFC(16, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 5, nn.LeakyReLU(0.2, True)),
        )

        self.encoder3 = nn.Sequential(
            ResBlockFC(8, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 16, nn.LeakyReLU(0.2, True)),
        )
        self.decoder3 = nn.Sequential(
            ResBlockFC(16, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 32, nn.LeakyReLU(0.2, True)),
            ResBlockFC(32, 8, nn.LeakyReLU(0.2, True)),
        )

    def forward(self, x):
        x1 = x[:, :11]
        x2 = x[:, 11:66].reshape(-1, 11, 5).reshape(-1, 5)
        x3 = x[:, 66:]
        z1 = self.encoder1(x1)
        y1 = self.decoder1(z1)
        z2 = self.encoder2(x2)
        y2 = self.decoder2(z2).reshape(-1, 11, 5)
        z3 = self.encoder3(x3)
        y3 = self.decoder3(z3)
        y = torch.cat([y1, y2.reshape(-1, 55), y3], dim=1)
        return y


class AEcoderResConv(nn.Module):
    def __init__(self, nc=128):
        super(AEcoderResConv, self).__init__()
        self.encoder12 = nn.Sequential(
            ResBlockConv( 6, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            nn.AdaptiveAvgPool1d(6),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc, 16, nn.LeakyReLU(0.2, True)),

            nn.Conv1d(16, 16, 6, 1, 0, bias=True)
        )
        self.decoder12 = nn.Sequential(
            nn.ConvTranspose1d(16, 16, 6, 1, 0, bias=False),
            nn.LeakyReLU(0.2, True),

            ResBlockConv(16, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            nn.AdaptiveAvgPool1d(11),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockConv(nc,  6, nn.LeakyReLU(0.2, True)),
        )

        self.encoder3 = nn.Sequential(
            ResBlockFC(8, nc, nn.LeakyReLU(0.2, True)),
            ResBlockFC(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockFC(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockFC(nc, 16, nn.LeakyReLU(0.2, True)),
        )
        self.decoder3 = nn.Sequential(
            ResBlockFC(16, nc, nn.LeakyReLU(0.2, True)),
            ResBlockFC(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockFC(nc, nc, nn.LeakyReLU(0.2, True)),
            ResBlockFC(nc, 8, nn.LeakyReLU(0.2, True)),
        )

    def forward(self, x):
        x1 = x[:, :11].reshape(-1, 1, 11)
        x2 = x[:, 11:66].reshape(-1, 11, 5).transpose(dim0=1, dim1=2)
        x12 = torch.cat([x1, x2], dim=1)
        x3 = x[:, 66:]
        z12 = self.encoder12(x12)
        y12 = self.decoder12(z12)

        z3 = self.encoder3(x3)
        y3 = self.decoder3(z3)
        y = torch.cat([y12[:, 0].reshape(-1, 11), y12[:, 1:].reshape(-1, 55), y3], dim=1)
        return y


class HookPhyNetFC(nn.Module):
    def __init__(self, in_dim: int, features: Sequence[int] = (128, 128, 128, 128)):
        super(HookPhyNetFC, self).__init__()
        layers = []
        pre_dim = in_dim
        for cur_dim in features:
            layers.append(ResBlockFC(pre_dim, cur_dim))
            pre_dim = cur_dim
        # layers.append(nn.Dropout(0.1))
        self.feature_net = nn.Sequential(*layers)
        self.fit = nn.Linear(features[-1], 2, bias=True)
        # self.fit = LinearDropConnect(features[-1], 2, bias=True, p=0.1)

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def ema_update(self, other, eta):
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            self_param.data.mul_(eta).add_(other_param.data, alpha=1 - eta)

    def forward(self, x):
        fea = self.feature_net(x)
        return torch.abs(self.fit(fea))


class LinearDropConnect(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.5, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.bernoulli(torch.ones_like(self.weight) * (1 - self.p))
            return F.linear(x, self.weight * mask / (1 - self.p), self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, p={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.p,
        )