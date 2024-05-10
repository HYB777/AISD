import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class TrainerCLDMP:
    def __init__(self, net, train_loader, val_loader, test_loader, opt, scale_dict):
        super(TrainerCLDMP, self).__init__()
        opt['scale'] = scale_dict
        self.net = net
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optim.Adam(self.net.parameters(), lr=opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=opt['lr'], momentum=0.9, weight_decay=5e-4)
        self.lr_schedule = optim.lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.9999)

        self.init()

        self.fit_loss = nn.L1Loss()

        self.best_loss = np.inf
        self.best_vm_rmae = -1
        self.best_area_rmae = -1
        self.best_epoch = -1
        self.best_acc = 0
        self.best_mat = None

    def init(self):

        print('-------------------------------------')
        print(self.net)
        print('-------------------------------------')
        if self.opt['cuda']:
            self.net.cuda()

        if self.opt['checkpoints_weight'] != '':
            check = torch.load('%s/%s' % (self.opt['checkpoints'], self.opt['checkpoints_weight']))
            self.net.load_state_dict(check['net_state_dict'])
            # self.optimizer.load_state_dict(check['optimizer_net_state_dict'])

    def train_epoch(self, epoch):

        train_bar = tqdm(self.train_loader)
        total_loss = 0
        cl_rmae = 0
        cd_rmae = 0
        cm_rmae = 0
        cp_rmae = 0
        self.net.train()
        for X, a, cl, cd, cm, cp in train_bar:
            self.optimizer.zero_grad()

            if self.opt['cuda']:
                X = X.cuda()
                a = a.cuda()
                cl = cl.cuda()
                cd = cd.cuda()
                cm = cm.cuda()
                cp = cp.cuda()

            z1, z2, z3, z4 = self.net(X, a)
            # print(z1.shape, cl.shape, z2.shape, cd.shape)
            loss = (1 - self.opt['regular']) * (self.fit_loss(z1.reshape(-1), cl.reshape(-1)) + self.fit_loss(z2.reshape(-1), cd.reshape(-1))) \
                   + self.opt['regular'] * (self.fit_loss(z3.reshape(-1), cm.reshape(-1)) + self.fit_loss(z4.reshape(-1), cp.reshape(-1)))
            rmae1 = (torch.abs(cl.reshape(-1) - z1.reshape(-1)).sum() / torch.abs(cl).sum())
            rmae2 = (torch.abs(cd.reshape(-1) - z2.reshape(-1)).sum() / torch.abs(cd).sum())
            rmae3 = (torch.abs(cm.reshape(-1) - z3.reshape(-1)).sum() / torch.abs(cm).sum())
            rmae4 = (torch.abs(cp.reshape(-1) - z4.reshape(-1)).sum() / torch.abs(cp).sum())

            total_loss += loss.item()
            cl_rmae += rmae1.item()
            cd_rmae += rmae2.item()
            cm_rmae += rmae3.item()
            cp_rmae += rmae4.item()
            loss.backward()
            self.optimizer.step()
            self.lr_schedule.step()

            train_bar.set_description(desc='[%d/%d] loss: %.4e,  cl_rmae: %.4e, cd_rmae: %.4e, cm_rmae: %.4e, cp_rmae: %.4e'
                                           % (epoch, self.opt['nEpoch'], loss.item(), rmae1.item(), rmae2.item(), rmae3.item(), rmae4.item()))

        print('\n[%d/%d] train phase | loss: %.4e,  cl_rmae: %.4e,  cd_rmae: %.4e,  cm_rmae: %.4e,  cp_rmae: %.4e'
              % (epoch,
                 self.opt['nEpoch'],
                 total_loss / len(self.train_loader),
                 cl_rmae / len(self.train_loader),
                 cd_rmae / len(self.train_loader),
                 cm_rmae / len(self.train_loader),
                 cp_rmae / len(self.train_loader),
                 )
              )

        return total_loss / len(self.train_loader), cl_rmae / len(self.train_loader), cd_rmae / len(self.train_loader), cm_rmae / len(self.train_loader), cp_rmae / len(self.train_loader)

    def val_epoch(self, epoch):
        self.net.eval()
        with torch.no_grad():
            val_bar = tqdm(self.val_loader)
            total_loss = 0

            cl_rmae_u = 0
            cl_rmae_d = 0
            cd_rmae_u = 0
            cd_rmae_d = 0
            cm_rmae_u = 0
            cm_rmae_d = 0
            cp_rmae_u = 0
            cp_rmae_d = 0

            for X, a, cl, cd, cm, cp in val_bar:

                if self.opt['cuda']:
                    X = X.cuda()
                    a = a.cuda()
                    cl = cl.cuda()
                    cd = cd.cuda()
                    cm = cm.cuda()
                    cp = cp.cuda()

                z1, z2, z3, z4 = self.net(X, a)
                loss = (1 - self.opt['regular']) * (
                            self.fit_loss(z1.reshape(-1), cl.reshape(-1)) + self.fit_loss(z2.reshape(-1),
                                                                                          cd.reshape(-1))) \
                       + self.opt['regular'] * (
                                   self.fit_loss(z3.reshape(-1), cm.reshape(-1)) + self.fit_loss(z4.reshape(-1),
                                                                                                 cp.reshape(-1)))
                rmae1 = (torch.abs(cl.reshape(-1) - z1.reshape(-1)).sum() / torch.abs(cl).sum())
                rmae2 = (torch.abs(cd.reshape(-1) - z2.reshape(-1)).sum() / torch.abs(cd).sum())
                rmae3 = (torch.abs(cm.reshape(-1) - z3.reshape(-1)).sum() / torch.abs(cm).sum())
                rmae4 = (torch.abs(cp.reshape(-1) - z4.reshape(-1)).sum() / torch.abs(cp).sum())

                cl_rmae_u += torch.abs(cl.reshape(-1) - z1.reshape(-1)).sum().item()
                cl_rmae_d += torch.abs(cl).sum().item()
                cd_rmae_u += torch.abs(cd.reshape(-1) - z2.reshape(-1)).sum().item()
                cd_rmae_d += torch.abs(cd).sum().item()
                cm_rmae_u += torch.abs(cm.reshape(-1) - z3.reshape(-1)).sum().item()
                cm_rmae_d += torch.abs(cm).sum().item()
                cp_rmae_u += torch.abs(cp.reshape(-1) - z4.reshape(-1)).sum().item()
                cp_rmae_d += torch.abs(cp).sum().item()

                total_loss += loss.item()

                val_bar.set_description(desc='[%d/%d] loss: %.4e,  cl_rmae: %.4e, cd_rmae: %.4e, cm_rmae: %.4e, cp_rmae: %.4e'
                                           % (epoch, self.opt['nEpoch'], loss.item(), rmae1.item(), rmae2.item(), rmae3.item(), rmae4.item()))

            print('\n[%d/%d] val phase | loss: %.4e,  cl_rmae: %.4e,  cd_rmae: %.4e,  cm_rmae: %.4e,  cp_rmae: %.4e'
              % (epoch,
                 self.opt['nEpoch'],
                 total_loss / len(self.val_loader),
                 cl_rmae_u / cl_rmae_d,
                 cd_rmae_u / cd_rmae_d,
                 cm_rmae_u / cm_rmae_d,
                 cp_rmae_u / cp_rmae_d,
                 )
              )

            mean_mae = total_loss / len(self.val_loader)
            mean_rmae1 = cl_rmae_u / cl_rmae_d
            mean_rmae2 = cd_rmae_u / cd_rmae_d
            if mean_rmae1 + mean_rmae2 < self.best_loss:
                self.best_loss = mean_rmae1 + mean_rmae2
                self.best_vm_rmae = mean_rmae1
                self.best_area_rmae = mean_rmae2
                self.best_epoch = epoch
                checkpoint_best = {
                    'epoch': epoch,
                    'loss': mean_mae,
                    'cl_rmae': mean_rmae1,
                    'cd_rmae': mean_rmae2,
                    'cm_rmae': cm_rmae_u / cm_rmae_d,
                    'cp_rmae': cp_rmae_u / cp_rmae_d,
                    'net_state_dict': self.net.state_dict(),
                    'optimizer_net_state_dict': self.optimizer.state_dict(),
                    'option': self.opt
                }
                torch.save(checkpoint_best, '%s/net_best.tar' % self.opt['checkpoints'])
        return mean_mae, mean_rmae1, mean_rmae2, cm_rmae_u / cm_rmae_d, cp_rmae_u / cp_rmae_d

    def test(self):
        with torch.no_grad():
            input('%s/net_best.tar' % self.opt['checkpoints'])
            check = torch.load('%s/net_best.tar' % self.opt['checkpoints'])
            self.net.load_state_dict(check['net_state_dict'])
            self.net.eval()

            test_bar = tqdm(self.test_loader)
            total_loss = 0
            vm_rmae = 0
            area_rmae = 0
            rmae1_u = 0
            rmae2_u = 0
            rmae1_d = 0
            rmae2_d = 0

            for X, a, cl, cd, _, _ in test_bar:

                if self.opt['cuda']:
                    X = X.cuda()
                    a = a.cuda()
                    cl = cl.cuda()
                    cd = cd.cuda()

                z1, z2, _, _ = self.net(X, a)

                loss = self.fit_loss(z1.reshape(-1), cl.reshape(-1)) + self.fit_loss(z2.reshape(-1), cd.reshape(-1))
                rmae1 = (torch.abs(cl.reshape(-1) - z1.reshape(-1)).sum() / torch.abs(cl).sum())
                rmae2 = (torch.abs(cd.reshape(-1) - z2.reshape(-1)).sum() / torch.abs(cd).sum())
                rmae1_u += torch.abs(cl.reshape(-1) - z1.reshape(-1)).sum().item()
                rmae1_d += torch.abs(cl).sum().item()
                rmae2_u += torch.abs(cd.reshape(-1) - z2.reshape(-1)).sum().item()
                rmae2_d += torch.abs(cd).sum().item()
                total_loss += loss.item()
                vm_rmae += rmae1.item()
                area_rmae += rmae2.item()

                total_loss += loss.item()
                vm_rmae += rmae1.item()
                area_rmae += rmae2.item()

                test_bar.set_description(desc='loss: %.4e,  cl_rmae: %.4e,  cd_rmae: %.4e'
                                              % (loss.item(), rmae1.item(), rmae2.item()))

            print('testing phase: loss: %.4e  cl_rmae: %.4e  cd_rmae: %.4e' % (total_loss / len(self.test_loader),
                                                                               rmae1_u / rmae1_d,
                                                                               rmae2_u / rmae2_d))

    def run(self):
        train_loss_hist = []
        val_loss_hist = []

        train_cl_rmae_hist = []
        train_cd_rmae_hist = []
        train_cm_rmae_hist = []
        train_cp_rmae_hist = []

        val_cl_rmae_hist = []
        val_cd_rmae_hist = []
        val_cm_rmae_hist = []
        val_cp_rmae_hist = []

        for epoch in range(self.opt['nEpoch']):
            print('---------------------%s---------------------------' % self.opt['checkpoints'])
            train_loss, train_cl_rmae, train_cd_rmae, train_cm_rmae, train_cp_rmae = self.train_epoch(epoch)
            val_loss, val_cl_rmae, val_cd_rmae, val_cm_rmae, val_cp_rmae = self.val_epoch(epoch)
            print('************************************************************************')
            print('******best epoch: %d, cl rmae: %f, cd rmae: %f******\n' % (self.best_epoch,
                                                                                self.best_vm_rmae,
                                                                                self.best_area_rmae))
            print(self.best_mat)
            print('=========================================================================')

            train_loss_hist.append(train_loss)
            train_cl_rmae_hist.append(train_cl_rmae)
            train_cd_rmae_hist.append(train_cd_rmae)
            train_cm_rmae_hist.append(train_cm_rmae)
            train_cp_rmae_hist.append(train_cp_rmae)

            val_loss_hist.append(val_loss)
            val_cl_rmae_hist.append(val_cl_rmae)
            val_cd_rmae_hist.append(val_cd_rmae)
            val_cm_rmae_hist.append(val_cm_rmae)
            val_cp_rmae_hist.append(val_cp_rmae)

            x = np.linspace(0, len(train_loss_hist) - 1, len(train_loss_hist))

            plt.clf()
            plt.plot(x, train_loss_hist, 'r-', label='train loss')
            plt.plot(x, val_loss_hist, 'g-', label='val loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('%s/loss.jpg' % self.opt['checkpoints'])

            plt.clf()
            plt.plot(x, train_cl_rmae_hist, 'r-', label='train cl rmae')
            plt.plot(x, val_cl_rmae_hist, 'g-', label='val cl rmae')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('cl rmae')
            plt.savefig('%s/cl_rmae.jpg' % self.opt['checkpoints'])

            plt.clf()
            plt.plot(x, train_cd_rmae_hist, 'r-', label='train cd rmae')
            plt.plot(x, val_cd_rmae_hist, 'g-', label='val cd rmae')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('cd rmae')
            plt.savefig('%s/cd_rmae.jpg' % self.opt['checkpoints'])

            plt.clf()
            plt.plot(x, train_cm_rmae_hist, 'r-', label='train cm rmae')
            plt.plot(x, val_cm_rmae_hist, 'g-', label='val cm rmae')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('cm rmae')
            plt.savefig('%s/cm_rmae.jpg' % self.opt['checkpoints'])

            plt.clf()
            plt.plot(x, train_cp_rmae_hist, 'r-', label='train cp rmae')
            plt.plot(x, val_cp_rmae_hist, 'g-', label='val cp rmae')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('cp rmae')
            plt.savefig('%s/cp_rmae.jpg' % self.opt['checkpoints'])

        self.test()


class TrainerAE:
    def __init__(self, ae, train_loader, val_loader, test_loader, opt, mu, sig):
        super(TrainerAE, self).__init__()
        self.ae = ae
        self.mu = mu
        self.sig = sig
        self.x_controls = np.hstack([0, (0.5 * (1 - np.cos(np.pi * np.arange(34 - 1).astype(np.double) / (34 - 2))))])
        opt['mu'] = mu
        opt['sig'] = sig

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

    def init(self):

        print('--------------AE-----------------------')
        print(self.ae)

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
            recon_x = self.ae(real_x)

            recon_xn = recon_x * self.sig.cuda() + self.mu.cuda()
            g_regular = (
                self.relu(-recon_xn[:, 0] + recon_xn[:, 1]).mean() +
                self.relu(-recon_xn[:, 0]).mean() +
                self.relu(recon_xn[:, 1, 0]).mean() +
                torch.abs(recon_xn[:, :, 1:] - recon_xn[:, :, :-1]).mean() +
                torch.abs(recon_xn[:, :, 0]).mean() + torch.abs(recon_xn[:, :, -1]).mean()
            )

            if self.g_phase % 20 == 0:
                for i in range(10):
                    plt.figure(100 + i)
                    example_real = real_x[i].detach().cpu()
                    example_recon = recon_x[i].detach().cpu()
                    example_real = (example_real * self.sig + self.mu).reshape(-1)
                    example_recon = (example_recon * self.sig + self.mu).reshape(-1)

                    example_real_upper = np.hstack([0, example_real[:32].numpy(), 0])
                    example_real_lower = np.hstack([0, example_real[32:].numpy(), 0])
                    plt.plot(self.x_controls, example_real_upper, 'ro-', label='real upper')
                    plt.plot(self.x_controls, example_real_lower, 'ro-', label='real lower')
                    example_recon_upper = np.hstack([0, example_recon[:32].numpy(), 0])
                    example_recon_lower = np.hstack([0, example_recon[32:].numpy(), 0])
                    plt.plot(self.x_controls, example_recon_upper, 'go--', label='recon upper')
                    plt.plot(self.x_controls, example_recon_lower, 'go--', label='recon lower')
                    plt.legend()
                    plt.axis('equal')
                    plt.savefig('%s/example_%d.png' % (self.opt['checkpoints'], i))
                    plt.clf()
                    plt.close(100 + i)

            loss = torch.abs(real_x - recon_x).mean() + 0.1 * g_regular

            total_loss += loss.item()
            g_loss += g_regular.item()
            loss.backward()
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
        self.ae.load_state_dict(check['net_state_dict'])
        self.ae.eval()
        test_bar = tqdm(self.test_loader)
        total_loss = 0
        err = []
        for real_x in test_bar:

            real_x = real_x.cuda()
            real_x = real_x.cuda()
            recon_x = self.ae(real_x)
            loss = torch.abs(real_x - recon_x).mean()
            err.append(loss.item())
            total_loss += loss.item()

            test_bar.set_description(desc='loss: %.4e' % (loss.item()))

        print('\n test phase | loss: %.4e' % (total_loss / len(self.test_loader)))
        print('%.4e, %.4e' % (np.mean(err), np.std(err)**2))
        return np.mean(err), np.std(err)**2

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


class TrainerGAN:
    def __init__(self, netD, netG, train_loader, val_loader, test_loader, opt, mu, sig, max_, min_):
        super(TrainerGAN, self).__init__()
        self.x_controls = np.hstack([0, (0.5 * (1 - np.cos(np.pi * np.arange(34 - 1).astype(np.double) / (34 - 2))))])

        self.netD = netD
        self.netG = netG
        opt['mu'] = mu.item()
        opt['sig'] = sig.item()
        self.opt = opt
        self.mu = mu
        self.sig = sig
        self.min_ = min_
        self.max_ = max_

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)

        self.REAL = 1.0
        self.FAKE = 0.0

        self.init()

        self.criterion = nn.BCEWithLogitsLoss()

        self.best_loss = np.inf
        self.best_acc = 0
        self.best_epoch = -1
        self.best_mat = None
        self.g_phase = 0

        self.FAIR = torch.from_numpy(np.load('config/RRFy.npy')).float().cuda()

        self.relu = nn.ReLU(True)

    def compute_mmd(self, real_samples, fake_samples, theta=0.1):
        bs = real_samples.shape[0]
        real_samples = real_samples.reshape(bs, -1)
        fake_samples = fake_samples.reshape(bs, -1)

        real_real = (real_samples.unsqueeze(1) - real_samples.unsqueeze(0)).norm(dim=-1)
        fake_fake = (fake_samples.unsqueeze(1) - fake_samples.unsqueeze(0)).norm(dim=-1)
        real_fake = (real_samples.unsqueeze(1) - fake_samples.unsqueeze(0)).norm(dim=-1)
        krr = torch.exp(-real_real**2 / (2 * theta ** 2)).mean()
        kff = torch.exp(-fake_fake**2 / (2 * theta ** 2)).mean()
        krf = torch.exp(-real_fake**2 / (2 * theta ** 2)).mean()
        return krr + kff - 2 * krf

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.netD(interpolates)
        # print(real_samples.shape, fake_samples.shape, interpolates.shape, d_interpolates.shape)
        fake = torch.ones(real_samples.shape[0], 1).cuda()
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def init(self):

        print('--------------discriminator-----------------------')
        print(self.netD)
        print('-----------------generator--------------------')
        print(self.netG)
        print('-----------------------------------------------')
        if self.opt['cuda']:
            self.netD.cuda()
            self.netG.cuda()

        if self.opt['checkpoints_weight'] != '':
            check = torch.load('%s/%s' % (self.opt['checkpoints'], self.opt['checkpoints_weight']))
            self.netD.load_state_dict(check['netD_state_dict'])
            self.netG.load_state_dict(check['netG_state_dict'])

    def train_epoch(self, epoch):

        train_bar = tqdm(self.train_loader)
        total_d_loss = 0
        total_g_loss = 0
        total_mmd = 0
        total_gp = 0
        acc_d = 0
        acc_g = 0
        regular_loss = 0

        self.netD.train()
        self.netG.train()
        for real_x in train_bar:
            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()
            self.g_phase += 1
            b_size = real_x.size(0)
            real_y = torch.full((b_size,), self.REAL, dtype=torch.float).cuda()
            fake_y = torch.full((b_size,), self.FAKE, dtype=torch.float).cuda()

            # D phase  ------------------------------------------------------------------------------------
            self.optimizerD.zero_grad()

            real_x = real_x.cuda()
            noise = torch.randn(b_size, self.opt['embed_dim']).cuda()
            fake_x = self.netG(noise)
            total_mmd += self.compute_mmd(real_x.data, fake_x.detach().data).item()

            d_real = self.netD(real_x).view(-1)
            d_fake = self.netD(fake_x.detach()).view(-1)
            gp = self.compute_gradient_penalty(real_x.data, fake_x.detach().data)
            acc_d_ = ((d_real > 0).sum().float() / b_size + (d_fake < 0).sum().float() / b_size) / 2
            # d_smooth = (d_real**2).mean()

            if self.opt['lambda_gp'] > 0:

                if self.opt['wass']:
                    d_loss = (d_fake - d_real).mean() + self.opt['lambda_gp'] * gp
                else:
                    d_loss = self.criterion(d_real, real_y) + self.criterion(d_fake, fake_y) + self.opt['lambda_gp'] * gp
            else:
                if self.opt['wass']:
                    d_loss = (d_fake - d_real).mean() + self.opt['lambda_gp'] * gp
                else:
                    d_loss = self.criterion(d_real, real_y) + self.criterion(d_fake, fake_y)
            total_d_loss += d_loss.item()
            total_gp += gp.item()
            d_loss.backward()
            self.optimizerD.step()
            acc_d += acc_d_.item()
            # G phase  ------------------------------------------------------------------------------------
            self.optimizerG.zero_grad()
            noise = torch.randn(b_size, self.opt['embed_dim']).cuda()
            fake_x = self.netG(noise)
            pfake = torch.nn.functional.pad(fake_x, [1, 1], value=0)
            g_regular = (
                self.relu(-fake_x[:, 0] + fake_x[:, 1]).mean() +
                self.relu(-fake_x[:, 0]).mean() +
                self.relu(fake_x[:, 1, 0]).mean() +
                self.relu(fake_x[:, 0] - self.max_).mean() +
                self.relu(-fake_x[:, 1] + self.min_).mean() +
                torch.abs(fake_x[:, :, 1:] - fake_x[:, :, :-1]).mean() +
                torch.abs(fake_x[:, :, 0]).mean() + torch.abs(fake_x[:, :, -1]).mean() +
                (pfake.unsqueeze(-2) @ self.FAIR.unsqueeze(0).unsqueeze(0) @ pfake.unsqueeze(-1)).mean() * 0.5
            )

            if self.g_phase % 100 == 0:
                if self.g_phase % 1000 == 0:
                    checkpoint_best = {
                        'epoch': epoch,
                        'mmd': total_mmd / len(self.train_loader),
                        'netD_state_dict': self.netD.state_dict(),
                        'netG_state_dict': self.netG.state_dict(),
                        'optimizerD_net_state_dict': self.optimizerD.state_dict(),
                        'optimizerG_net_state_dict': self.optimizerG.state_dict(),
                        'option': self.opt
                    }
                    torch.save(checkpoint_best, '%s/net_best_draw.tar' % self.opt['checkpoints'])

                for i in range(10):
                    pic = 'example_%d.png' % i if self.opt['regular'] == 0 else 'example_regular_%d.png' % i
                    plt.figure(100 + i)

                    example = fake_x[i].detach().cpu().reshape(-1)
                    # example = example * self.sig + self.mu
                    example_upper = np.hstack([0, example[:32].numpy(), 0])
                    example_lower = np.hstack([0, example[32:].numpy(), 0])
                    plt.plot(self.x_controls, example_upper, 'ro-', label='real upper')
                    plt.plot(self.x_controls, example_lower, 'go-', label='real lower')
                    plt.legend()
                    plt.axis('equal')
                    plt.savefig(self.opt['checkpoints'] + '/' + pic)
                    plt.clf()
                    plt.close(100 + i)
            output = self.netD(fake_x).view(-1)  # b*2*64
            acc_g_ = (output > 0).sum().float() / b_size
            if self.opt['regular'] > 0:
                if self.opt['wass']:
                    g_loss = (-output).mean() + self.opt['regular'] * g_regular
                else:
                    g_loss = self.criterion(output, real_y) + self.opt['regular'] * g_regular
            else:
                if self.opt['wass']:
                    g_loss = (-output).mean()
                else:
                    g_loss = self.criterion(output, real_y)
            # print(g_loss.shape, g_regular.shape, fake_x.shape, output.shape)
            total_g_loss += g_loss.item() - self.opt['regular'] * g_regular.item()
            regular_loss += g_regular.item()
            g_loss.backward()
            self.optimizerG.step()
            acc_g += acc_g_.item()
            train_bar.set_description(desc='[%d/%d] D loss: %.4e  G loss: %.4e  GP: %.4e Regular : %.4e' % (epoch, self.opt['nEpoch'],
                                                                                             d_loss.item(),
                                                                                             g_loss.item(),
                                                                                             gp.item(), g_regular.item()))

        print('\n[%d/%d] train phase | D loss: %.4e  G loss: %.4e  GP: %.4e  MMD: %.4e  ACC D: %.4f ACC G: %.4f, Regular: %.4e'
              % (epoch, self.opt['nEpoch'],
                 total_d_loss / len(self.train_loader),
                 total_g_loss / len(self.train_loader),
                 total_gp / len(self.train_loader),
                 total_mmd / len(self.train_loader),
                 acc_d / len(self.train_loader),
                 acc_g / len(self.train_loader),
                 regular_loss / len(self.train_loader)))
        mean_mmd_loss = total_mmd / len(self.train_loader)

        if mean_mmd_loss < self.best_loss:
            self.best_loss = mean_mmd_loss
            self.best_epoch = epoch
            checkpoint_best = {
                'epoch': epoch,
                'mmd': mean_mmd_loss,
                'netD_state_dict': self.netD.state_dict(),
                'netG_state_dict': self.netG.state_dict(),
                'optimizerD_net_state_dict': self.optimizerD.state_dict(),
                'optimizerG_net_state_dict': self.optimizerG.state_dict(),
                'option': self.opt
            }
            torch.save(checkpoint_best, '%s/net_bestG.tar' % self.opt['checkpoints'])

        return (total_d_loss - self.opt['lambda_gp'] * total_gp) / len(self.train_loader),\
               total_g_loss / len(self.train_loader), \
               total_gp / len(self.train_loader), \
               total_mmd / len(self.train_loader)

    def val_epoch(self, epoch):
        self.netD.eval()
        self.netG.eval()

        val_bar = tqdm(self.val_loader)
        total_d_loss = 0
        acc = 0
        for real_x in val_bar:
            b_size = real_x.size(0)
            real_y = torch.full((b_size,), self.REAL, dtype=torch.float).cuda()

            # D phase  ------------------------------------------------------------------------------------

            real_x = real_x.cuda()

            d_real = self.netD(real_x).view(-1)
            d_loss = self.criterion(d_real, real_y)
            total_d_loss += d_loss.item()
            acc_ = (d_real > 0).sum().float() / b_size

            val_bar.set_description(desc='[%d/%d] D loss: %.4e  acc: %.4f' % (epoch, self.opt['nEpoch'],
                                                                                             d_loss.item(),
                                                                                             acc_.item()))
            acc += acc_.item()

        print('\n[%d/%d] val phase | D loss: %.4e  Acc: %.4f'
              % (epoch, self.opt['nEpoch'],
                 total_d_loss / len(self.val_loader),
                 acc / len(self.val_loader)))

        mean_d_loss = total_d_loss / len(self.val_loader)
        mean_acc = acc / len(self.val_loader)

        if mean_acc >= self.best_acc:
            self.best_acc = mean_acc
            self.best_epoch = epoch

            checkpoint_best = {
                'epoch': epoch,
                'acc': mean_acc,
                'netD_state_dict': self.netD.state_dict(),
                'netG_state_dict': self.netG.state_dict(),
                'optimizerD_net_state_dict': self.optimizerD.state_dict(),
                'optimizerG_net_state_dict': self.optimizerG.state_dict(),
                'option': self.opt
            }
            torch.save(checkpoint_best, '%s/net_bestD.tar' % self.opt['checkpoints'])

        return mean_d_loss, mean_acc

    def test(self):
        check = torch.load('%s/net_best.tar' % self.opt['checkpoints'])
        self.netD.load_state_dict(check['netD_state_dict'])
        self.netG.load_state_dict(check['netG_state_dict'])
        self.netD.eval()
        self.netG.eval()
        test_bar = tqdm(self.test_loader)
        total_d_loss = 0
        total_g_loss = 0
        total_mmd = 0
        total_gp = 0
        for real_x in test_bar:
            b_size = real_x.size(0)
            real_y = torch.full((b_size,), self.REAL, dtype=torch.float).cuda()
            fake_y = torch.full((b_size,), self.FAKE, dtype=torch.float).cuda()

            # D phase  ------------------------------------------------------------------------------------

            real_x = real_x.cuda()
            noise = torch.randn(b_size, self.opt['embed_dim']).cuda()
            fake_x = self.netG(noise)
            total_mmd += self.compute_mmd(real_x.data, fake_x.detach().data).item()

            d_real = self.netD(real_x).view(-1)
            d_fake = self.netD(fake_x.detach()).view(-1)
            gp = self.compute_gradient_penalty(real_x.data, fake_x.detach().data)
            if self.opt['lambda_gp'] > 0:
                d_loss = self.criterion(d_real, real_y) + self.criterion(d_fake, fake_y) + self.opt['lambda_gp'] * gp
            else:
                d_loss = self.criterion(d_real, real_y) + self.criterion(d_fake, fake_y)
            total_d_loss += d_loss.item()
            total_gp += gp.item()

            # G phase  ------------------------------------------------------------------------------------
            output = self.netD(fake_x).view(-1)
            g_loss = self.criterion(output, real_y)
            total_g_loss += g_loss.item()

            test_bar.set_description(desc='D loss: %.4e  G loss: %.4e  GP: %.4e'
                                          % (d_loss.item(), g_loss.item(), gp.item()))

        print('\n test phase | D loss: %.4e  G loss: %.4e  GP: %.4e  MMD: %.4e'
              % (total_d_loss / len(self.test_loader),
                 total_g_loss / len(self.test_loader),
                 total_gp / len(self.test_loader),
                 total_mmd / len(self.test_loader)))

    def run(self):
        train_d_loss_hist = []
        train_g_loss_hist = []
        train_gp_loss_hist = []
        train_mmd_loss_hist = []

        val_d_loss_hist = []
        val_acc_hist = []

        for epoch in range(self.opt['nEpoch']):
            print('---------------------%s---------------------------' % self.opt['checkpoints'])
            train_d_loss, train_g_loss, train_gp_loss, train_mmd_loss = self.train_epoch(epoch)
            val_d_loss, val_acc = self.val_epoch(epoch)
            print('************************************************************************')
            print('******best epoch: %d, mmd: %f******\n' % (self.best_epoch, self.best_loss))
            print('=========================================================================')

            train_d_loss_hist.append(train_d_loss)
            train_g_loss_hist.append(train_g_loss)
            train_gp_loss_hist.append(train_gp_loss)
            train_mmd_loss_hist.append(train_mmd_loss)

            val_d_loss_hist.append(val_d_loss)
            val_acc_hist.append(val_acc)

            x = np.linspace(0, len(train_d_loss_hist) - 1, len(train_d_loss_hist))

            plt.clf()
            plt.plot(x, train_d_loss_hist, 'r-', label='train d loss')
            plt.plot(x, train_g_loss_hist, 'g-', label='train g loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('%s/gan_loss.jpg' % self.opt['checkpoints'])

            plt.clf()
            plt.plot(x, val_acc_hist, 'g-', label='val acc')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.savefig('%s/G_acc.jpg' % self.opt['checkpoints'])

            plt.clf()
            plt.plot(x, train_gp_loss_hist, 'r-', label='train gp loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('%s/GP_loss.jpg' % self.opt['checkpoints'])

            plt.clf()
            plt.plot(x, train_mmd_loss_hist, 'r-', label='train mmd loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('%s/MMD_loss.jpg' % self.opt['checkpoints'])

        # self.test()