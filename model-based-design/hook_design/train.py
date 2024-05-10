import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class TrainerAE:
    def __init__(self, ae, train_loader, val_loader, test_loader, opt, mu, sig):
        super(TrainerAE, self).__init__()
        opt['mu'] = mu
        opt['sig'] = sig
        self.mu = torch.from_numpy(mu).float()
        self.sig = torch.from_numpy(sig).float()

        self.ae = ae

        self.opt = opt

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optim.Adam(self.ae.parameters(), lr=opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)

        self.init()

        self.criterion = nn.L1Loss()

        self.best_loss = np.inf
        self.best_epoch = -1

        self.relu = nn.ReLU(True)

        self.g_phase = 0

    def init(self):

        # print('--------------AE-----------------------')
        # print(self.ae)
        #
        # print('-----------------------------------------------')
        if self.opt['cuda']:
            self.ae.cuda()

        if self.opt['checkpoints_weight'] != '':
            check = torch.load('%s/%s' % (self.opt['checkpoints'], self.opt['checkpoints_weight']))
            self.ae.load_state_dict(check['net_state_dict'])

    def train_epoch(self, epoch):
        self.g_phase = self.g_phase + 1
        draw_flag = False

        train_bar = tqdm(self.train_loader)
        total_loss = 0
        g_loss = 0

        self.ae.train()
        for real_x in train_bar:
            self.optimizer.zero_grad()

            real_x = real_x.cuda()
            recon_x = self.ae(real_x)

            if self.g_phase % 20 == 0 and not draw_flag:
                for i in range(10):
                    plt.figure(100 + i)
                    example_real = real_x[i].detach().cpu()
                    example_recon = recon_x[i].detach().cpu()

                    plt.scatter(example_real, example_recon, label='real vs recon')
                    plt.legend()
                    plt.axis('equal')
                    plt.savefig('%s/example_%d.png' % (self.opt['checkpoints'], i))
                    plt.clf()
                    plt.close(100 + i)
                    draw_flag = True

            loss = torch.abs(real_x - recon_x).mean()

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            train_bar.set_description(desc='[%d/%d] loss: %.4e' % (epoch, self.opt['nEpoch'], loss.item()))

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
        self.ae.load_state_dict(check['net_state_dict'])
        self.ae.eval()
        test_bar = tqdm(self.test_loader)
        total_loss = 0
        loss_hist = []
        cnt = 0
        for real_x in test_bar:

            real_x = real_x.cuda()
            recon_x = self.ae(real_x)
            loss = torch.abs(real_x - recon_x).mean()
            loss_hist.append(loss.item())
            total_loss += loss.item()
            # print(cnt, loss.item())
            # if cnt == 817:
            #     print(real_x)
            #     print(loss.item())
            #     input('wait')
            cnt += 1
            test_bar.set_description(desc='loss: %.4e' % (loss.item()))

        print('\n test phase | loss: %.4e' % (total_loss / len(self.test_loader)))
        loss_hist = np.array(loss_hist)
        print('mean: ', loss_hist.mean(), 'std: ', loss_hist.std())

        return loss_hist.mean(), loss_hist.std()**2

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


