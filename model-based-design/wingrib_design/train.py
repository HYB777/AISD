import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class TrainerAE:
    def __init__(self, ae, train_loader, val_loader, test_loader, opt):
        super(TrainerAE, self).__init__()
        self.airfoil_u = np.loadtxt('config/naca4415_upper.txt')
        self.airfoil_l = np.loadtxt('config/naca4415_lower.txt')
        self.spars0_pts = np.loadtxt('config/spars0.txt')
        self.spars1_pts = np.loadtxt('config/spars1.txt')
        self.ae = ae

        self.mu = torch.from_numpy(opt['mu']).float()
        self.sig = torch.from_numpy(opt['sig']).float()

        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optim.Adam(self.ae.parameters(), lr=opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)

        self.init()

        self.criterion = nn.L1Loss()

        self.best_loss = np.inf
        self.best_epoch = -1
        self.best_mat = None
        self.g_phase = 0

        self.relu = nn.ReLU(True)

        A = np.eye(16)
        A[1:, :-1] = A[1:, :-1] - np.eye(15)
        A[0, -1] = -1
        self.A = torch.from_numpy(A).float().cuda()

    def init(self):

        print('--------------AE-----------------------')
        # print(self.ae)

        print('-----------------------------------------------')
        if self.opt['cuda']:
            self.ae.cuda()

        if self.opt['checkpoints_weight'] != '':
            check = torch.load('%s/%s' % (self.opt['checkpoints'], self.opt['checkpoints_weight']))
            self.ae.load_state_dict(check['net_state_dict'])

    def train_epoch(self, epoch):

        train_bar = tqdm(self.train_loader)
        total_loss = 0
        g_loss = 0

        self.ae.train()
        for real_x in train_bar:
            self.g_phase += 1
            self.optimizer.zero_grad()
            self.g_phase += 1

            real_x = real_x.cuda()
            recon_x = self.ae(real_x)  # (B, 3, 3, 2, 16)
            recon_xn = recon_x * self.sig.cuda() + self.mu.cuda()
            g_regular = torch.abs(recon_xn@self.A).sum((1, 2)).mean()
            if self.g_phase % 20 == 0:
                for i in range(10):
                    fig_id = 310
                    example_real = (real_x * self.sig.cuda() + self.mu.cuda())[i].detach().cpu()
                    example_recon = recon_xn[i].detach().cpu()
                    for j in range(3):
                        plt.subplot(fig_id + j + 1)

                        plt.plot(example_real[j, 0, 0], example_real[j, 0, 1], 'ro-')
                        plt.plot(example_recon[j, 0, 0], example_recon[j, 0, 1], 'go-')
                        plt.plot(example_real[j, 1, 0], example_real[j, 1, 1], 'ro-')
                        plt.plot(example_recon[j, 1, 0], example_recon[j, 1, 1], 'go-')
                        plt.plot(example_real[j, 2, 0], example_real[j, 2, 1], 'ro-')
                        plt.plot(example_recon[j, 2, 0], example_recon[j, 2, 1], 'go-')

                        plt.plot(self.airfoil_l[:, 0], self.airfoil_l[:, 1], 'k-')
                        plt.plot(self.airfoil_u[:, 0], self.airfoil_u[:, 1], 'k-')
                        plt.fill(self.spars0_pts[:, 0], self.spars0_pts[:, 1], 'b')
                        plt.fill(self.spars1_pts[:, 0], self.spars1_pts[:, 1], 'b')
                        plt.axis('equal')
                    plt.savefig('%s/example_%d.png' % (self.opt['checkpoints'], i))
                    plt.clf()
                    plt.close()

            loss = torch.abs(real_x - recon_x).mean() + 0.1 * g_regular

            total_loss += loss.item()
            g_loss += g_regular.item()
            loss.backward()
            # for p in self.ae.parameters():
            #     print(torch.sum(p.grad**2)**0.5)
            self.optimizer.step()

            train_bar.set_description(desc='[%d/%d] loss: %.4e g regular: %.4e' % (epoch, self.opt['nEpoch'], loss.item(), g_regular.item()))

        print('\n[%d/%d] train phase | loss: %.4e, g loss: %.4e'
              % (epoch, self.opt['nEpoch'],
                 total_loss / len(self.train_loader), g_loss / len(self.train_loader)))
        mean_loss = total_loss / len(self.train_loader)

        return mean_loss

    def val_epoch(self, epoch):
        self.ae.eval()
        with torch.no_grad():

            val_bar = tqdm(self.val_loader)
            total_loss = 0
            for real_x in val_bar:

                real_x = real_x.cuda()
                recon_x = self.ae(real_x)
                loss = torch.abs(real_x - recon_x).mean()
                total_loss += loss.item()
                val_bar.set_description(desc='[%d/%d] loss: %.4e' % (epoch, self.opt['nEpoch'], loss.item()))

            print('\n[%d/%d] val phase | loss: %.4e' % (epoch, self.opt['nEpoch'], total_loss / len(self.val_loader)))
            mean_loss = total_loss / len(self.val_loader)

            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.best_epoch = epoch
                checkpoint_best = {
                    'epoch': epoch,
                    'mae': mean_loss,
                    'net_state_dict': self.ae.state_dict(),
                    'optimizerD_net_state_dict': self.optimizer.state_dict(),
                    'option': self.opt
                }
                torch.save(checkpoint_best, '%s/ae_best_zeros.tar' % self.opt['checkpoints'])

            return mean_loss

    def test(self):
        check = torch.load('%s/ae_best_zeros.tar' % self.opt['checkpoints'])
        # check = torch.load('learning_results/results_ae_ngf*ndf=32*32/ae_best_zeros.tar')
        # input()
        self.ae.load_state_dict(check['net_state_dict'])
        self.ae.eval()
        test_bar = tqdm(self.test_loader)
        total_loss = 0
        e_list = []
        for real_x in test_bar:

            real_x = real_x.cuda()
            recon_x = self.ae(real_x)
            loss = torch.abs(real_x - recon_x).mean()
            total_loss += loss.item()
            e_list.append(loss.item())
            test_bar.set_description(desc='loss: %.4e' % (loss.item()))
        print(np.mean(e_list), np.std(e_list)**2)
        print('\n test phase | loss: %.4e' % (total_loss / len(self.test_loader)))
        return np.mean(e_list), np.std(e_list)**2

    def run(self):
        train_loss_hist = []
        val_loss_hist = []

        for epoch in range(self.opt['nEpoch']):
            print('---------------------%s---------------------------' % self.opt['checkpoints'])
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            print('************************************************************************')
            print('******best epoch: %d, mae: %f******\n' % (self.best_epoch, self.best_loss))
            print('=========================================================================')

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            x = np.linspace(0, len(train_loss_hist) - 1, len(train_loss_hist))

            plt.clf()
            plt.plot(x, train_loss_hist, 'r-', label='train loss')
            plt.plot(x, val_loss_hist, 'g-', label='val loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('%s/ae_loss.jpg' % self.opt['checkpoints'])

        self.test()