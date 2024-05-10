import torch
from torch import nn
import torch.nn.functional as F


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
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
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
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.freq_net = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1, bias=True)
        )
        self.vm_net = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1, bias=True)
        )
        self.area_net = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1, bias=True)
        )

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        out5 = self.avg_pool(out4)
        out6 = torch.flatten(out5, 1)

        freq = self.freq_net(out6)
        vm = self.vm_net(out6)
        area = self.area_net(out6)

        return torch.abs(freq), torch.abs(vm), torch.abs(area)


class ResNet1d18FitOne(nn.Module):
    def __init__(self):
        super(ResNet1d18FitOne, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
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
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1, bias=True)
        )

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        out5 = self.avg_pool(out4)
        out6 = torch.flatten(out5, 1)

        out7 = self.fc(out6)

        return out7


class ResNet1d18FitThreeNet(nn.Module):
    def __init__(self):
        super(ResNet1d18FitThreeNet, self).__init__()
        self.freq_net = ResNet1d18FitOne()
        self.vm_net = ResNet1d18FitOne()
        self.area_net = ResNet1d18FitOne()

    def forward(self, x):
        # (N, 1, 64, 64)
        freq = self.freq_net(x)
        vm = self.vm_net(x)
        area = self.area_net(x)

        return torch.abs(freq), torch.abs(vm), torch.abs(area)


class AEcoder(nn.Module):
    def __init__(self, embed_dim, ndf, ngf):
        super(AEcoder, self).__init__()
        self.encoder = Encoder(ndf, embed_dim)
        self.decoder = Decoder(ngf, embed_dim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class Encoder(nn.Module):
    def __init__(self, ndf, embed_dim):
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
        self.conv5 = nn.Conv1d(ndf * 2, embed_dim, 2, 1, 0, bias=False)

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
