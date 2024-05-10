import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from torch.autograd.functional import jacobian
from nets import *

N_PARAMS = 14
NCOLS = 150
MARGIN = 1.0  # / 20_0000 / 2.0
WEIGHT = 0.0
DECAY_RATE = 0.9999
DECAY_STEP = 150


class TrainerFitLL:
    def __init__(self, collector, opt, mu_paras, sig_paras, semi_supervised=True):
        super(TrainerFitLL, self).__init__()
        opt['ct_var'] = 0
        opt['cq_var'] = 0
        opt['mu_paras'] = mu_paras
        opt['sig_paras'] = sig_paras

        self.semi_supervised = semi_supervised

        self.opt = opt
        self.collector = collector

        self.student = ResNet1d18_Rotor()
        self.teacher = ResNet1d18_Rotor()

        # self.optimizer = optim.Adam(self.student.parameters(), lr=opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)
        # self.lr_schedule = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150])
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.opt['lr'], betas=(0.9, 0.999), weight_decay=5e-5)
        # self.lr_schedule = optim.lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.9999)

        self.init()

        self.fit_loss = nn.L1Loss()

        self.best_loss = {'student': np.inf, 'teacher': np.inf}
        self.best_ct_rmae = {'student': -1., 'teacher': -1.}
        self.best_cq_rmae = {'student': -1., 'teacher': -1.}
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
        # l1 = 0.5*torch.abs(y)**2
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
        w_consistency = 1e-5  # * np.exp(-5*(1-epoch/self.opt['nEpoch'])**2)
        alpha = 0.99 if self.global_iter % 1000 <= milestones else 0.999
        return w_consistency, alpha

    def train_epoch(self, epoch):
        blocks, teacher_blocks, student_blocks = self.collector.train_blocks(10000)

        train_bar = tqdm(zip(blocks, teacher_blocks, student_blocks), ncols=NCOLS)
        total_loss = 0
        ct_rmae = 0
        cq_rmae = 0
        consistency_loss_ = 0
        self.student.train()
        self.teacher.train()

        w, alpha = self.weight_(epoch)

        for block, teacher_block, student_block in train_bar:
            X, n_blade, pitches, ct, cq = self.collector.dataset[block]
            teacher_x_noise, _, teacher_pitches_noise, _, _ = self.collector.dataset[teacher_block]
            student_x_noise, _, student_pitches_noise, _, _ = self.collector.dataset[student_block]
            is_labeled = torch.as_tensor(self.collector.is_labeled[block])
            w_teacher_noise = torch.rand(len(X)) * 0.1 * (1 - is_labeled)
            w_student_noise = torch.rand(len(X)) * 0.1 * (1 - is_labeled)
            # print(X.shape, paras.shape, w_student_noise.shape)
            X_teacher = (1 - w_teacher_noise.view(-1, 1, 1)) * X + w_teacher_noise.view(-1, 1, 1) * teacher_x_noise
            pitches_teacher = (1 - w_teacher_noise.view(-1, 1)) * pitches + w_teacher_noise.view(-1, 1) * teacher_pitches_noise
            X_student = (1 - w_student_noise.view(-1, 1, 1)) * X + w_student_noise.view(-1, 1, 1) * student_x_noise
            pitches_student = (1 - w_student_noise.view(-1, 1)) * pitches + w_student_noise.view(-1, 1) * student_pitches_noise
            num_labeled = sum(is_labeled)
            num_unlabeled = len(is_labeled) - num_labeled

            self.optimizer.zero_grad()

            if self.opt['cuda']:
                X_teacher = X_teacher.cuda()
                pitches_teacher = pitches_teacher.cuda()
                X_student = X_student.cuda()
                pitches_student = pitches_student.cuda()
                ct = ct.cuda()
                cq = cq.cuda()
                is_labeled = is_labeled.cuda()
                n_blade = n_blade.cuda()
            y = torch.cat([ct.view(-1, 1), cq.view(-1, 1)], dim=1)
            y_hat_student = self.student(X_student, n_blade, pitches_student)
            y_hat_teacher = self.teacher(X_teacher, n_blade, pitches_teacher).detach()
            # print(y.shape, y_hat_student.shape)

            regress_loss = (self.l1_loss(y-y_hat_student, reduction='keepdim') * is_labeled).sum() / num_labeled
            consistency_loss = (self.l1_loss(y_hat_teacher-y_hat_student, reduction='keepdim') * (1 - is_labeled)).sum() / num_unlabeled
            loss = regress_loss + w * consistency_loss

            ct_rmae_ = (torch.abs(y[:, 0] - y_hat_student[:, 0]).sum() / torch.abs(y[:, 0]).sum())
            cq_rmae_ = (torch.abs(y[:, 1] - y_hat_student[:, 1]).sum() / torch.abs(y[:, 1]).sum())

            total_loss += loss.item()
            ct_rmae += ct_rmae_.item()
            cq_rmae += cq_rmae_.item()
            consistency_loss_ += consistency_loss.item()

            loss.backward()
            self.optimizer.step()
            self.teacher.ema_update(self.student, alpha)
            self.global_iter = self.global_iter + 1

            train_bar.set_description(desc='[%d/%d] train | loss: %.4e ct=%.4e cq=%.4e step: %d size: %d #labeled: %d, consis: %.4e'
                                           % (epoch, self.opt['nEpoch'], loss.item(),
                                              ct_rmae_.item(), cq_rmae_.item(),
                                              self.global_iter, self.collector.data_size(), num_labeled, consistency_loss.item()))

        print('[%d/%d] train phase | loss=%.4e, ct=%.4e, cq=%.4e, consistency_loss=%.4e'
              % (epoch, self.opt['nEpoch'],
                 total_loss / len(blocks),
                 ct_rmae / len(blocks),
                 cq_rmae / len(blocks),
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
        ct_rmae = 0
        cq_rmae = 0
        self.student.train()
        self.teacher.train()

        _, alpha = self.weight_(epoch)

        for X, n_blade, pitches, ct, cq, is_labeled in train_bar:
            assert torch.all(is_labeled)
            if self.opt['cuda']:
                X = X.cuda()
                ct = ct.cuda()
                cq = cq.cuda()
                n_blade = n_blade.cuda()
                pitches = pitches.cuda()

            self.optimizer.zero_grad()

            y = torch.cat([ct.view(-1, 1), cq.view(-1, 1)], dim=1)
            y_hat_student = self.student(X, n_blade, pitches)
            loss = self.l1_loss(y-y_hat_student, reduction='mean')

            ct_rmae_ = (torch.abs(y[:, 0] - y_hat_student[:, 0]).sum() / torch.abs(y[:, 0]).sum())
            cq_rmae_ = (torch.abs(y[:, 1] - y_hat_student[:, 1]).sum() / torch.abs(y[:, 1]).sum())

            total_loss += loss.item()
            ct_rmae += ct_rmae_.item()
            cq_rmae += cq_rmae_.item()

            loss.backward()
            self.optimizer.step()
            self.teacher.ema_update(self.student, alpha)
            self.global_iter = self.global_iter + 1

            train_bar.set_description(desc='[%d/%d] train | loss: %.4e cl=%.4e cd=%.4e step: %d size: %d'
                                           % (epoch, self.opt['nEpoch'], loss.item(),
                                              ct_rmae_.item(), cq_rmae_.item(),
                                              self.global_iter, self.collector.data_size()))

        print('[%d/%d] train phase | loss=%.4e, ct=%.4e, cq=%.4e'
              % (epoch, self.opt['nEpoch'],
                 total_loss / len(train_loader),
                 ct_rmae / len(train_loader),
                 cq_rmae / len(train_loader)))

    def val_epoch(self, epoch):
        self.student.eval()
        self.teacher.eval()
        self.collector.eval()
        val_loader = torch.utils.data.DataLoader(dataset=self.collector, num_workers=self.opt['workers'],
                                                 batch_size=10000, shuffle=False, drop_last=False,
                                                 pin_memory=False, worker_init_fn=self.opt['_init_fn'])
        val_bar = tqdm(val_loader)
        total_loss = {'student': 0., 'teacher': 0.}

        ct_rmae_u = {'student': 0., 'teacher': 0.}
        ct_rmae_d = {'student': 0., 'teacher': 0.}
        cq_rmae_u = {'student': 0., 'teacher': 0.}
        cq_rmae_d = {'student': 0., 'teacher': 0.}

        keys = ['student', 'teacher']
        for X, n_blade, pitches, ct, cq, is_labeled in val_bar:
            assert torch.all(is_labeled)
            if self.opt['cuda']:
                X = X.cuda()
                pitches = pitches.cuda()
                ct = ct.cuda()
                cq = cq.cuda()
                n_blade = n_blade.cuda()

            y = torch.cat([ct.view(-1, 1), cq.view(-1, 1)], dim=1)
            y_hat_dict = {'teacher': self.teacher(X, n_blade, pitches), 'student': self.student(X, n_blade, pitches)}
            ct_rmae_ = {'student': 0., 'teacher': 0.}
            cq_rmae_ = {'student': 0., 'teacher': 0.}
            loss_item = {'student': 0., 'teacher': 0.}
            for k in keys:
                y_hat = y_hat_dict[k]
                loss = self.fit_loss(y, y_hat)

                loss_item[k] = loss.item()
                ct_rmae_[k] = (torch.abs(y[:, 0] - y_hat[:, 0]).sum() / torch.abs(y[:, 0]).sum())
                cq_rmae_[k] = (torch.abs(y[:, 1] - y_hat[:, 1]).sum() / torch.abs(y[:, 1]).sum())

                ct_rmae_u[k] += torch.abs(y[:, 0] - y_hat[:, 0]).sum().item()
                ct_rmae_d[k] += torch.abs(y[:, 0]).sum().item()
                cq_rmae_u[k] += torch.abs(y[:, 1] - y_hat[:, 1]).sum().item()
                cq_rmae_d[k] += torch.abs(y[:, 1]).sum().item()

                total_loss[k] += loss.item()

            val_bar.set_description(desc='[%d/%d] val | '
                                         'student-[loss=%.4e, ct=%.4e, cq=%.4e], '
                                         'teacher=[loss=%.4e, ct=%.4e, cq=%.4e]'
                                         % (epoch, self.opt['nEpoch'],
                                            loss_item['student'], ct_rmae_['student'], cq_rmae_['student'],
                                            loss_item['teacher'], ct_rmae_['teacher'], cq_rmae_['teacher'],))

        print('[%d/%d] val phase |'
              'student-[loss=%.4e, cl=%.4e, cd=%.4e], '
              'teacher=[loss=%.4e, cl=%.4e, cd=%.4e]'
              % (epoch, self.opt['nEpoch'],
                 total_loss['student'] / len(val_loader),
                 ct_rmae_u['student'] / ct_rmae_d['student'],
                 cq_rmae_u['student'] / cq_rmae_d['student'],

                 total_loss['teacher'] / len(val_loader),
                 ct_rmae_u['teacher'] / ct_rmae_d['teacher'],
                 cq_rmae_u['teacher'] / cq_rmae_d['teacher'])
              )

        for k in keys:
            mean_mae = total_loss[k] / len(val_loader)
            mean_ct_rmae = ct_rmae_u[k] / ct_rmae_d[k]
            mean_cq_rmae = cq_rmae_u[k] / cq_rmae_d[k]
            if mean_ct_rmae + mean_cq_rmae < self.best_loss[k]:
                self.convergence_flag = 0
                self.best_loss[k] = mean_ct_rmae + mean_cq_rmae
                self.best_ct_rmae[k] = mean_ct_rmae
                self.best_cq_rmae[k] = mean_cq_rmae
                self.best_epoch[k] = epoch
                checkpoint_best = {
                    'epoch': epoch,
                    'loss': mean_mae,
                    'ct_rmae': mean_ct_rmae,
                    'cq_rmae': mean_cq_rmae,
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
            for X, n_blade, pitches, ind, _, teacher_data_noise, teacher_pitches_noise, student_data_noise, student_pitches_noise in bar:

                if self.opt['cuda']:
                    X = X.cuda()
                    n_blade = n_blade.cuda()
                    pitches = pitches.cuda()
                    teacher_data_noise = teacher_data_noise.cuda()
                    teacher_pitches_noise = teacher_pitches_noise.cuda()
                    student_data_noise = student_data_noise.cuda()
                    student_pitches_noise = student_pitches_noise.cuda()
                eps_teacher = torch.rand(len(X), device=X.device) * 0.1
                eps_student = torch.rand(len(X), device=X.device) * 0.1
                y_hat_student = self.student(
                    (1 - eps_student.view(-1, 1, 1)) * X + eps_student.view(-1, 1, 1) * student_data_noise,
                    n_blade,
                    (1 - eps_student.view(-1, 1)) * pitches + eps_student.view(-1, 1) * student_pitches_noise
                )  # (bs, dim_out)
                y_hat_teacher = self.teacher(
                    (1 - eps_teacher.view(-1, 1, 1)) * X + eps_teacher.view(-1, 1, 1) * teacher_data_noise,
                    n_blade,
                    (1 - eps_teacher.view(-1, 1)) * pitches + eps_teacher.view(-1, 1) * teacher_pitches_noise
                )  # (bs, dim_out)
                # y_hat_student = self.student(X)  # (bs, dim_out)
                # y_hat_teacher = self.teacher(X)  # (bs, dim_out)
                err = torch.abs(y_hat_student - y_hat_teacher)

                ind2uncertainty.update_val(ind.detach().numpy().reshape(-1),
                                           err.detach().cpu().numpy(), n_blade.detach().cpu().numpy())
        r_ind, r_uncertainty = ind2uncertainty.err_mean()
        return r_ind, r_uncertainty

    def update_best_score(self):
        try:
            student = ResNet1d18_Rotor()
            teacher = ResNet1d18_Rotor()

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

        student.eval()
        teacher.eval()
        self.collector.eval()
        val_loader = torch.utils.data.DataLoader(dataset=self.collector, num_workers=self.opt['workers'],
                                                 batch_size=1000, shuffle=False, drop_last=False,
                                                 pin_memory=False, worker_init_fn=self.opt['_init_fn'])
        val_bar = tqdm(val_loader)
        ct_rmae_u = {'student': 0., 'teacher': 0.}
        ct_rmae_d = {'student': 0., 'teacher': 0.}
        cq_rmae_u = {'student': 0., 'teacher': 0.}
        cq_rmae_d = {'student': 0., 'teacher': 0.}
        keys = ['student', 'teacher']
        with torch.no_grad():
            for X, n_blade, pitches, ct, cq, is_labeled in val_bar:
                assert torch.all(is_labeled)
                if self.opt['cuda']:
                    X = X.cuda()
                    n_blade = n_blade.cuda()
                    pitches = pitches.cuda()
                    ct = ct.cuda()
                    cq = cq.cuda()

                y = torch.cat([ct.view(-1, 1), cq.view(-1, 1)], dim=1)
                y_hat_dict = {'teacher': teacher(X, n_blade, pitches), 'student': student(X, n_blade, pitches)}
                for k in keys:
                    y_hat = y_hat_dict[k]

                    ct_rmae_u[k] += torch.abs(y[:, 0] - y_hat[:, 0]).sum().item()
                    ct_rmae_d[k] += torch.abs(y[:, 0]).sum().item()
                    cq_rmae_u[k] += torch.abs(y[:, 1] - y_hat[:, 1]).sum().item()
                    cq_rmae_d[k] += torch.abs(y[:, 1]).sum().item()

            print('update phase |'
                  'student-[cl=%.4e, cd=%.4e], '
                  'teacher=[cl=%.4e, cd=%.4e]'
                  % (ct_rmae_u['student'] / ct_rmae_d['student'], cq_rmae_u['student'] / cq_rmae_d['student'],
                     ct_rmae_u['teacher'] / ct_rmae_d['teacher'], cq_rmae_u['teacher'] / cq_rmae_d['teacher']))
            for k in keys:
                mean_ct_rmae = ct_rmae_u[k] / ct_rmae_d[k]
                mean_cq_rmae = cq_rmae_u[k] / cq_rmae_d[k]
                self.best_loss[k] = mean_ct_rmae + mean_cq_rmae
                self.best_ct_rmae[k] = mean_ct_rmae
                self.best_cq_rmae[k] = mean_cq_rmae

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
            print('ct rmae: ', self.best_ct_rmae)
            print('cq rmae: ', self.best_cq_rmae)
            print('convergence: ', self.convergence_flag)
            print('=========================================================================')
            epoch = epoch + 1


# 19549580
# opt fix + lr decay ： se_rmae: 4.9883e-03  vm_rmae: 2.6518e-03, 38
# opt fix ：            se_rmae: 4.7776e-03  vm_rmae: 2.5242e-03, 30 *
# opt update：          se_rmae: 4.7440e-03  vm_rmae: 2.5287e-03, 31 *
# opt update + lr decay ： se_rmae: 4.7987-03  vm_rmae: 2.6641-03, 30


class BatchDict:
    def __init__(self, ):
        self.ind_dict = {}

    def update_val(self, keys, vals, n_blade):
        """
        :param keys: (bs, )
        :param vals: (bs, dim_out)
        :param n_blade: (bs, )
        :return:
        """
        for k, v, nb in zip(keys, vals, n_blade):
            if nb not in self.ind_dict.keys():
                self.ind_dict[nb] = {}
            if k not in self.ind_dict[nb].keys():
                self.ind_dict[nb][k] = []
            self.ind_dict[nb][k].append(v)

    def err_mean(self):
        ks = {}
        mean_err = {}
        for nb in self.ind_dict.keys():
            ks_nb = []
            mean_err_nb = []
            for k in tqdm(self.ind_dict[nb].keys()):
                ks_nb.append(k)
                mean_err_nb.append(np.array(self.ind_dict[nb][k]).mean())

            ks_nb = np.array(ks_nb)  # (S, )
            mean_err_nb = np.array(mean_err_nb)  # (S, )
            ks[nb] = ks_nb
            mean_err[nb] = mean_err_nb

        return ks, mean_err
