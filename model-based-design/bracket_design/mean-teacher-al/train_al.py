import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from torch.autograd.functional import jacobian
from nets import *

NCOLS = 150
MARGIN = 1.0  # / 20_0000 / 2.0
WEIGHT = 0.0
DECAY_RATE = 0.9999
DECAY_STEP = 150


class TrainerFitLL:
    def __init__(self, collector, opt, mu_paras, sig_paras, semi_supervised=True):
        super(TrainerFitLL, self).__init__()
        opt['freq_var'] = 0
        opt['vm_var'] = 0
        opt['area_var'] = 0
        opt['mu_paras'] = mu_paras
        opt['sig_paras'] = sig_paras

        self.semi_supervised = semi_supervised

        self.opt = opt
        self.collector = collector

        self.student = ResNet1d18Fit()
        self.teacher = ResNet1d18Fit()

        # self.optimizer = optim.Adam(self.student.parameters(), lr=opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)
        # self.lr_schedule = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150])
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)
        # self.lr_schedule = optim.lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.9999)

        self.init()

        self.fit_loss = nn.L1Loss()

        self.best_loss = {'student': np.inf, 'teacher': np.inf}
        self.best_freq_rmae = {'student': -1., 'teacher': -1.}
        self.best_vm_rmae = {'student': -1., 'teacher': -1.}
        self.best_area_rmae = {'student': -1., 'teacher': -1.}
        self.best_epoch = {'student': -1., 'teacher': -1.}

        self.global_iter = 0
        self.data_size = 0
        self.bs = 128

        self.warmed = False

        self.convergence_flag = 0

    def nets_cuda(self):
        self.student.cuda()
        self.teacher.cuda()

    def init(self):

        print('-------------------------------------')
        print(self.student)
        print(self.teacher)
        print('-------------------------------------')
        if self.opt['cuda']:
            self.nets_cuda()

    def load_state_dict(self, check, no_collector=False):
        self.student.load_state_dict(check['student'])
        self.teacher.load_state_dict(check['teacher'])
        self.optimizer.load_state_dict(check['optimizer'])
        # self.lr_schedule.load_state_dict(check['lr_schedule'])

        self.global_iter = check['global_iter']
        self.best_loss = check['best_loss']
        self.warmed = check['warmed']
        if not no_collector:
            self.collector.load_state_dict(check['collector'])

    def state_dict(self):
        return {
            'warmed': self.warmed,
            'student': self.student.state_dict(),
            'teacher': self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'lr_schedule': self.lr_schedule.state_dict(),
            'global_iter': self.global_iter,
            'best_loss': self.best_loss,
            'collector': self.collector.state_dict()
        }

    def l1_loss(self, y, reduction='mean'):
        l1 = torch.abs(y)
        if len(l1.shape) == 2:
            l1 = l1.mean(1)
        if reduction == 'mean':
            return l1.mean()
        elif reduction == 'sum':
            return l1.sum()
        elif reduction == 'keepdim':
            return l1
        else:
            raise KeyError()

    def weight_(self, epoch):
        milestones = 700  # self.opt['nEpoch']//5 * 2.0
        # w_consistency = 1.  # min(epoch / milestones, 1.0) * 0.1,
        w_consistency = 0.0001  # * np.exp(-5*(1-epoch/self.opt['nEpoch'])**2)
        alpha = 0.99 if self.global_iter % 1000 <= milestones else 0.999
        return w_consistency, alpha

    def train_epoch(self, epoch):
        blocks, teacher_blocks, student_blocks = self.collector.train_blocks(1024)

        train_bar = tqdm(zip(blocks, teacher_blocks, student_blocks), ncols=NCOLS)
        total_loss = 0
        freq_rmae = 0
        vm_rmae = 0
        area_rmae = 0
        consistency_loss_ = 0
        self.student.train()
        self.teacher.train()

        w, alpha = self.weight_(epoch)

        for block, teacher_block, student_block in train_bar:
            X, freq, vm, area = self.collector.dataset[block]
            teacher_noise, _, _, _ = self.collector.dataset[teacher_block]
            student_noise, _, _, _ = self.collector.dataset[student_block]
            is_labeled = torch.as_tensor(self.collector.is_labeled[block])
            w_teacher_noise = torch.rand(len(X)) * 0.1 * (1 - is_labeled)
            w_student_noise = torch.rand(len(X)) * 0.1 * (1 - is_labeled)
            X_teacher = (1 - w_teacher_noise.view(-1, 1, 1)) * X + w_teacher_noise.view(-1, 1, 1) * teacher_noise
            X_student = (1 - w_student_noise.view(-1, 1, 1)) * X + w_student_noise.view(-1, 1, 1) * student_noise
            num_labeled = sum(is_labeled)
            num_unlabeled = len(is_labeled) - num_labeled

            self.optimizer.zero_grad()

            if self.opt['cuda']:
                X_teacher = X_teacher.cuda()
                X_student = X_student.cuda()
                freq = freq.cuda()
                vm = vm.cuda()
                area = area.cuda()
                is_labeled = is_labeled.cuda()
            y = torch.cat([freq.view(-1, 1), vm.view(-1, 1), area.view(-1, 1)], dim=1)
            y_hat_student = self.student(X_student)
            y_hat_teacher = self.teacher(X_teacher).detach()

            regress_loss = (self.l1_loss(y-y_hat_student, reduction='keepdim') * is_labeled).sum() / num_labeled
            consistency_loss = (self.l1_loss(y_hat_teacher-y_hat_student, reduction='keepdim') * (1 - is_labeled)).sum() / num_unlabeled
            loss = regress_loss + w * consistency_loss

            se_rmae_ = (torch.abs(y[:, 0] - y_hat_student[:, 0]).sum() / torch.abs(y[:, 0]).sum())
            vol_rmae_ = (torch.abs(y[:, 1] - y_hat_student[:, 1]).sum() / torch.abs(y[:, 1]).sum())
            area_rmae_ = (torch.abs(y[:, 2] - y_hat_student[:, 2]).sum() / torch.abs(y[:, 2]).sum())

            total_loss += loss.item()
            freq_rmae += se_rmae_.item()
            vm_rmae += vol_rmae_.item()
            area_rmae += area_rmae_.item()
            consistency_loss_ += consistency_loss.item()

            loss.backward()
            self.optimizer.step()
            self.teacher.ema_update(self.student, alpha)
            self.global_iter = self.global_iter + 1

            train_bar.set_description(desc='[%d/%d] train | loss: %.4e freq: %.4e vm: %.4e area: %.4e step: %d size: %d #labeled: %d, consis: %.4e'
                                           % (epoch, self.opt['nEpoch'], loss.item(),
                                              se_rmae_.item(), vol_rmae_.item(), area_rmae_.item(),
                                              self.global_iter, self.collector.data_size(), num_labeled, consistency_loss.item()))

        print('[%d/%d] train phase | loss=%.4e, freq=%.4e, vm=%.4e, area=%.4e, consistency_loss=%.4e'
              % (epoch, self.opt['nEpoch'],
                 total_loss / len(blocks),
                 freq_rmae / len(blocks),
                 vm_rmae / len(blocks),
                 area_rmae / len(blocks),
                 consistency_loss_ / len(blocks)))

    def train_epoch_only_labeled(self, epoch):
        self.student.train()
        self.teacher.train()
        self.collector.train()
        train_loader = torch.utils.data.DataLoader(dataset=self.collector, num_workers=self.opt['workers'],
                                                   batch_size=self.bs, shuffle=False, drop_last=False,
                                                   pin_memory=False, worker_init_fn=self.opt['_init_fn'])

        train_bar = tqdm(train_loader, ncols=NCOLS)
        total_loss = 0
        freq_rmae = 0
        vm_rmae = 0
        area_rmae = 0
        self.student.train()
        self.teacher.train()

        _, alpha = self.weight_(epoch)

        for X, freq, vm, area, is_labeled in train_bar:
            assert torch.all(is_labeled)
            if self.opt['cuda']:
                X = X.cuda()
                freq = freq.cuda()
                vm = vm.cuda()
                area = area.cuda()

            self.optimizer.zero_grad()

            y = torch.cat([freq.view(-1, 1), vm.view(-1, 1), area.view(-1, 1)], dim=1)
            y_hat_student = self.student(X)
            loss = self.l1_loss(y-y_hat_student, reduction='mean')

            freq_rmae_ = (torch.abs(y[:, 0] - y_hat_student[:, 0]).sum() / torch.abs(y[:, 0]).sum())
            vm_rmae_ = (torch.abs(y[:, 1] - y_hat_student[:, 1]).sum() / torch.abs(y[:, 1]).sum())
            area_rmae_ = (torch.abs(y[:, 2] - y_hat_student[:, 2]).sum() / torch.abs(y[:, 2]).sum())

            total_loss += loss.item()
            freq_rmae += freq_rmae_.item()
            vm_rmae += vm_rmae_.item()
            area_rmae += area_rmae_.item()

            loss.backward()
            self.optimizer.step()
            self.teacher.ema_update(self.student, alpha)
            self.global_iter = self.global_iter + 1

            train_bar.set_description(desc='[%d/%d] train | loss: %.4e freq: %.4e vm: %.4e area: %.4e step: %d size: %d'
                                           % (epoch, self.opt['nEpoch'], loss.item(),
                                              freq_rmae_.item(), vm_rmae_.item(), area_rmae_.item(),
                                              self.global_iter, self.collector.data_size()))

        print('[%d/%d] train phase | loss=%.4e, freq=%.4e, vm=%.4e, area=%.4e'
              % (epoch, self.opt['nEpoch'],
                 total_loss / len(train_loader),
                 freq_rmae / len(train_loader),
                 vm_rmae / len(train_loader),
                 area_rmae / len(train_loader)))

    def val_epoch(self, epoch):
        self.student.eval()
        self.teacher.eval()
        self.collector.eval()
        val_loader = torch.utils.data.DataLoader(dataset=self.collector, num_workers=self.opt['workers'],
                                                 batch_size=1000, shuffle=False, drop_last=False,
                                                 pin_memory=False, worker_init_fn=self.opt['_init_fn'])
        val_bar = tqdm(val_loader)
        total_loss = {'student': 0., 'teacher': 0.}
        freq_rmae_u = {'student': 0., 'teacher': 0.}
        freq_rmae_d = {'student': 0., 'teacher': 0.}
        vm_rmae_u = {'student': 0., 'teacher': 0.}
        vm_rmae_d = {'student': 0., 'teacher': 0.}
        area_rmae_u = {'student': 0., 'teacher': 0.}
        area_rmae_d = {'student': 0., 'teacher': 0.}
        keys = ['student', 'teacher']
        for X, freq, vm, area, is_labeled in val_bar:
            assert torch.all(is_labeled)
            if self.opt['cuda']:
                X = X.cuda()
                freq = freq.cuda()
                vm = vm.cuda()
                area = area.cuda()
            y = torch.cat([freq.view(-1, 1), vm.view(-1, 1), area.view(-1, 1)], dim=1)
            y_hat_dict = {'teacher': self.teacher(X), 'student': self.student(X)}
            freq_rmae_ = {'student': 0., 'teacher': 0.}
            vm_rmae_ = {'student': 0., 'teacher': 0.}
            area_rmae_ = {'student': 0., 'teacher': 0.}
            loss_item = {'student': 0., 'teacher': 0.}
            for k in keys:
                y_hat = y_hat_dict[k]
                loss = self.fit_loss(y, y_hat)

                loss_item[k] = loss.item()
                freq_rmae_[k] = (torch.abs(y[:, 0] - y_hat[:, 0]).sum() / torch.abs(y[:, 0]).sum())
                vm_rmae_[k] = (torch.abs(y[:, 1] - y_hat[:, 1]).sum() / torch.abs(y[:, 1]).sum())
                area_rmae_[k] = (torch.abs(y[:, 2] - y_hat[:, 2]).sum() / torch.abs(y[:, 2]).sum())

                freq_rmae_u[k] += torch.abs(y[:, 0] - y_hat[:, 0]).sum().item()
                freq_rmae_d[k] += torch.abs(y[:, 0]).sum().item()
                vm_rmae_u[k] += torch.abs(y[:, 1] - y_hat[:, 1]).sum().item()
                vm_rmae_d[k] += torch.abs(y[:, 1]).sum().item()
                area_rmae_u[k] += torch.abs(y[:, 2] - y_hat[:, 2]).sum().item()
                area_rmae_d[k] += torch.abs(y[:, 2]).sum().item()

                total_loss[k] += loss.item()

            val_bar.set_description(desc='[%d/%d] val | '
                                         'student-[loss=%.4e, freq=%.4e, vm=%.4e, area=%.4e], '
                                         'teacher=[loss=%.4e, freq=%.4e, vm=%.4e, area=%.4e]'
                                         % (epoch, self.opt['nEpoch'],
                                            loss_item['student'], freq_rmae_['student'], vm_rmae_['student'], area_rmae_['student'],
                                            loss_item['teacher'], freq_rmae_['teacher'], vm_rmae_['teacher'], area_rmae_['teacher'],))

        print('[%d/%d] val phase |'
              'student-[loss=%.4e, freq=%.4e, vm=%.4e, area=%.4e], '
              'teacher=[loss=%.4e, freq=%.4e, vm=%.4e, area=%.4e]'
              % (epoch, self.opt['nEpoch'],
                 total_loss['student'] / len(val_loader), freq_rmae_u['student'] / freq_rmae_d['student'],
                 vm_rmae_u['student'] / vm_rmae_d['student'], area_rmae_u['student'] / area_rmae_d['student'],

                 total_loss['teacher'] / len(val_loader), freq_rmae_u['teacher'] / freq_rmae_d['teacher'],
                 vm_rmae_u['teacher'] / vm_rmae_d['teacher'], area_rmae_u['teacher'] / area_rmae_d['teacher']))

        for k in keys:
            mean_mae = total_loss[k] / len(val_loader)
            mean_freq_rmae = freq_rmae_u[k] / freq_rmae_d[k]
            mean_vm_rmae = vm_rmae_u[k] / vm_rmae_d[k]
            mean_area_rmae = area_rmae_u[k] / area_rmae_d[k]
            if mean_freq_rmae + mean_vm_rmae + mean_area_rmae < self.best_loss[k]:
                self.best_loss[k] = mean_freq_rmae + mean_vm_rmae + mean_area_rmae
                self.best_freq_rmae[k] = mean_freq_rmae
                self.best_vm_rmae[k] = mean_vm_rmae
                self.best_area_rmae[k] = mean_area_rmae
                self.best_epoch[k] = epoch
                checkpoint_best = {
                    'epoch': epoch,
                    'loss': mean_mae,
                    'freq_rmae': mean_freq_rmae,
                    'vm_rmae': mean_vm_rmae,
                    'area_rmae': mean_area_rmae,
                    'student': self.student.state_dict(),
                    'teacher': self.teacher.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'option': self.opt,
                    'scales': self.collector.scales
                }
                torch.save(checkpoint_best, '%s/net_best_%s.tar' % (self.opt['checkpoints'], k))

    def sample_in_uncertainty(self, N=100, subset=-1):

        self.student.train()
        self.teacher.train()

        self.collector.pool(subset)
        test_loader = torch.utils.data.DataLoader(dataset=self.collector, num_workers=self.opt['workers'],
                                                  batch_size=1000, shuffle=True, drop_last=True,
                                                  pin_memory=False, worker_init_fn=self.opt['_init_fn'])

        ind2uncertainty = BatchDict()
        for t in range(N):
            bar = tqdm(test_loader, ncols=NCOLS)
            for X, ind, _, teacher_noise, student_noise in bar:

                if self.opt['cuda']:
                    X = X.cuda()
                    teacher_noise = teacher_noise.cuda()
                    student_noise = student_noise.cuda()
                eps_teacher = torch.rand(len(X), 1, 1, device=X.device) * 0.1
                eps_student = torch.rand(len(X), 1, 1, device=X.device) * 0.1
                y_hat_student = self.student((1 - eps_student) * X + eps_student * student_noise)  # (bs, dim_out)
                y_hat_teacher = self.teacher((1 - eps_teacher) * X + eps_teacher * teacher_noise)  # (bs, dim_out)
                # y_hat_student = self.student(X)  # (bs, dim_out)
                # y_hat_teacher = self.teacher(X)  # (bs, dim_out)
                err = torch.abs(y_hat_student - y_hat_teacher)

                ind2uncertainty.update_val(ind.detach().numpy().reshape(-1),
                                           err.detach().cpu().numpy())
        r_ind, r_uncertainty = ind2uncertainty.err_mean()
        return r_ind, r_uncertainty

    def update_best_score(self):
        try:
            student = ResNet1d18Fit()
            teacher = ResNet1d18Fit()

            check_student = torch.load('%s/net_best_student.tar' % (self.opt['checkpoints'],))
            check_teacher = torch.load('%s/net_best_teacher.tar' % (self.opt['checkpoints'],))

            student.cuda()
            student.load_state_dict(check_student['student'])

            teacher.cuda()
            teacher.load_state_dict(check_teacher['teacher'])

            print('loading best parameters')
        except Exception as e:
            print(e)
            student = copy(self.student)
            teacher = copy(self.teacher)

        student.train()
        teacher.train()
        self.collector.eval()
        val_loader = torch.utils.data.DataLoader(dataset=self.collector, num_workers=self.opt['workers'],
                                                 batch_size=1000, shuffle=False, drop_last=False,
                                                 pin_memory=False, worker_init_fn=self.opt['_init_fn'])
        val_bar = tqdm(val_loader)
        freq_rmae_u = {'student': 0., 'teacher': 0.}
        freq_rmae_d = {'student': 0., 'teacher': 0.}
        vm_rmae_u = {'student': 0., 'teacher': 0.}
        vm_rmae_d = {'student': 0., 'teacher': 0.}
        area_rmae_u = {'student': 0., 'teacher': 0.}
        area_rmae_d = {'student': 0., 'teacher': 0.}
        keys = ['student', 'teacher']
        with torch.no_grad():
            for X, freq, vm, area, is_labeled in val_bar:
                assert torch.all(is_labeled)
                if self.opt['cuda']:
                    X = X.cuda()
                    freq = freq.cuda()
                    vm = vm.cuda()
                    area = area.cuda()
                y = torch.cat([freq.view(-1, 1), vm.view(-1, 1), area.view(-1, 1)], dim=1)
                y_hat_dict = {'teacher': teacher(X), 'student': student(X)}
                for k in keys:
                    y_hat = y_hat_dict[k]

                    freq_rmae_u[k] += torch.abs(y[:, 0] - y_hat[:, 0]).sum().item()
                    freq_rmae_d[k] += torch.abs(y[:, 0]).sum().item()
                    vm_rmae_u[k] += torch.abs(y[:, 1] - y_hat[:, 1]).sum().item()
                    vm_rmae_d[k] += torch.abs(y[:, 1]).sum().item()
                    area_rmae_u[k] += torch.abs(y[:, 2] - y_hat[:, 2]).sum().item()
                    area_rmae_d[k] += torch.abs(y[:, 2]).sum().item()

            print('update phase |'
                  'student-[freq_rmae=%.4e, vm_rmae=%.4e, area_rmae=%.4e], '
                  'teacher=[freq_rmae=%.4e, vm_rmae=%.4e, area_rmae=%.4e]'
                  % (freq_rmae_u['student'] / freq_rmae_d['student'], vm_rmae_u['student'] / vm_rmae_d['student'],
                     area_rmae_u['student'] / area_rmae_d['student'],
                     freq_rmae_u['teacher'] / freq_rmae_d['teacher'], vm_rmae_u['teacher'] / vm_rmae_d['teacher'],
                     area_rmae_u['teacher'] / area_rmae_d['teacher'],))

            for k in keys:
                mean_freq_rmae = freq_rmae_u[k] / freq_rmae_d[k]
                mean_vm_rmae = vm_rmae_u[k] / vm_rmae_d[k]
                mean_area_rmae = area_rmae_u[k] / area_rmae_d[k]
                self.best_loss[k] = mean_freq_rmae + mean_vm_rmae + mean_area_rmae
                self.best_freq_rmae[k] = mean_freq_rmae
                self.best_vm_rmae[k] = mean_vm_rmae
                self.best_area_rmae[k] = mean_area_rmae

    def run(self, n_epochs, convergence_epoch=100):
        self.update_best_score()
        self.opt['nEpoch'] = n_epochs

        self.optimizer = optim.Adam(self.student.parameters(), lr=self.opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)

        self.convergence_flag = 0
        epoch = 0
        while epoch < n_epochs or self.convergence_flag < convergence_epoch:
            print('---------------------%s---------------------------' % self.opt['checkpoints'])
            self.convergence_flag = self.convergence_flag + 1
            if self.semi_supervised:
                self.train_epoch(epoch)
            else:
                self.train_epoch_only_labeled(epoch)
            # self.lr_schedule.step()
            with torch.no_grad():
                self.val_epoch(epoch)
            checkpoint_last = {
                'epoch': epoch,
                'student': self.student.state_dict(),
                'teacher': self.teacher.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'lr_schedule_state_dict': self.lr_schedule.state_dict(),
                'option': self.opt
            }
            torch.save(checkpoint_last, '%s/net_last.tar' % self.opt['checkpoints'])
            print('************************************************************************')
            print('best epoch:', self.best_epoch)
            print('freq rmae: ', self.best_freq_rmae)
            print('vm rmae: ', self.best_vm_rmae)
            print('area rmae: ', self.best_area_rmae)
            print('convergence: ', self.convergence_flag)
            print('=========================================================================')
            epoch = epoch + 1
            # if self.convergence_flag > 100:
            #     break


# 19549580
# opt fix + lr decay ： se_rmae: 4.9883e-03  vm_rmae: 2.6518e-03, 38
# opt fix ：            se_rmae: 4.7776e-03  vm_rmae: 2.5242e-03, 30 *
# opt update：          se_rmae: 4.7440e-03  vm_rmae: 2.5287e-03, 31 *
# opt update + lr decay ： se_rmae: 4.7987-03  vm_rmae: 2.6641-03, 30


class BatchDict:
    def __init__(self, ):
        self.ind_dict = {}
        self.cnt_dict = {}

    def update_val(self, keys, vals):
        """
        :param keys: (bs, )
        :param vals: (bs, dim_out)
        :return:
        """
        for k, v in zip(keys, vals):
            if k not in self.ind_dict.keys():
                self.ind_dict[k] = []
                self.cnt_dict[k] = 0
            self.cnt_dict[k] += 1
            self.ind_dict[k].append(v)

    def err_mean(self):
        ks = []
        mean_err = []
        for k in tqdm(self.ind_dict.keys()):
            ks.append(k)
            mean_err.append(np.array(self.ind_dict[k]).mean())

        ks = np.array(ks)         # (S, )
        mean_err = np.array(mean_err)     # (S, )

        return ks, mean_err

    def update(self, keys, vals, grads):
        """
        :param keys: (bs, )
        :param vals: (bs, dim_out)
        :param grads: (bs, dim_out, dim_in)
        :return:
        """
        for k, v, g in zip(keys, vals, grads):
            if k not in self.ind_dict.keys():
                self.ind_dict[k] = [[], []]
                self.cnt_dict[k] = 0
            self.cnt_dict[k] += 1
            self.ind_dict[k][0].append(v)
            self.ind_dict[k][1].append(g)

    def val_std_grad_svd(self):
        ks = []
        vals = []
        grads = []
        kmin = np.inf
        for k in self.cnt_dict.keys():
            kmin = min(kmin, self.cnt_dict[k])
        for k in self.ind_dict.keys():
            ks.append(k)

            vals.append(np.array(self.ind_dict[k][0][:kmin]))
            # (N, dim_out, dim_in) -> (dim_out, dim_in, N)
            grads_k = np.array(self.ind_dict[k][1][:kmin]).transpose((1, 2, 0))
            grads_norm_k = np.sum(grads_k**2, 1, keepdims=True)**0.5
            A = grads_k / (grads_norm_k + 1e-12)
            grads.append(A)

        ks = np.array(ks)         # (S, )
        vals = np.array(vals)     # (S, N, dim_out)
        grads = np.array(grads)   # (S, dim_out, dim_in, N) -> (S, dim_out, N)

        stds = vals.std(1).mean(1)
        sigmas = np.linalg.svd(grads)[1][:, :, 0]  # (S, dim_out)
        n = grads.shape[-1]
        svds = (1 - (sigmas**2 - 1) / (n - 1)).mean(1)  # (S, )
        return ks, stds + svds

    def val_std_grad_svd_(self):
        ks = []
        stds = []
        svds = []
        for k in tqdm(self.ind_dict.keys()):
            ks.append(k)
            stds.append(np.array(self.ind_dict[k][0]).std(0).mean())
            # (N, dim_out, dim_in) -> (dim_out, dim_in, N)
            grads_k = np.array(self.ind_dict[k][1]).transpose((1, 2, 0))
            grads_norm_k = np.sum(grads_k**2, 1, keepdims=True)**0.5
            A = grads_k / (grads_norm_k + 1e-12)
            n = grads_k.shape[-1]
            sigmas = np.linalg.svd(A)[1][:, 0]
            svds.append((1 - (sigmas ** 2 - 1) / (n - 1)).mean())

        ks = np.array(ks)         # (S, )
        stds = np.array(stds)     # (S, N, dim_out)
        svds = np.array(svds)   # (S, dim_out, dim_in, N) -> (S, dim_out, N)

        return ks, stds + svds
