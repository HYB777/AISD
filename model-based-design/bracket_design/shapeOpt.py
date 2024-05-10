import os

import matplotlib.pyplot as plt
import torch
from bspline_utils import BsplineArea
from models import ResNet1d18Fit
from model_ae import AEcoder
import numpy as np
from pyoptsparse import SLSQP, Optimization, PSQP, ALPSO, IPOPT
import scipy.io as scio
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--item', type=int, default=211, help='the data item')

seed = 0
random.seed(seed)
np.random.seed(seed)

CONFIG_BRACKET = {
    'lY': 0.072,
    'lZ': 0.070,
    'bDia': 0.008,
    'dCmp': 0.04,
    'rC': 0.004,
    'zC': 0.021,
    'sZ': 0.02,
}


AREA_LB = 0.0003167197452229302
MASS_UB = (0.20968 - AREA_LB * 7850 * 2.0 * 0.004)
AE_EPS = 0.006
MAX_ITER = 600
FREQ_LB = 60.
VM_UB = 90.
PENALTY = 0.0
RESULT_DIR = 'optimized_results/ALSSL_in_area_freq_%.2f_vm_%.2f' % (FREQ_LB, VM_UB)

try:
    os.mkdir(RESULT_DIR)
except OSError:
    pass


class BracketRequire:
    def __init__(self):
        super(BracketRequire, self).__init__()
        A = np.zeros((32, 33))
        A[:32, :32] = np.eye(32)
        A[0, -1] = 1
        self.A = torch.from_numpy(A).float()
        B = np.array([
            [-1, -1,  0,  0],
            [ 0,  0, -1, -1],
            [ 1,  0,  1,  0],
            [ 0,  1,  0,  1]
        ])
        self.B = torch.from_numpy(B).float()
        C = []
        for i in range(32):
            for j in range(i+2, 32):
                if i == 0 and j+1 == 32:
                    continue
                Cij = torch.zeros(1, 33, 4)
                Cij[0, i, 0] = 1
                Cij[0, i + 1, 1] = 1
                Cij[0, j, 2] = 1
                Cij[0, j + 1, 3] = 1
                C.append(Cij)
        self.C = torch.cat(C).float()

        Per = torch.eye(32)
        self.Per = torch.zeros((32, 32))
        self.Per[:16] = Per[16:]
        self.Per[16:] = Per[:16]

    def eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch_A = x_torch.reshape(1, 2, -1) @ self.A
        x_torch_seg = x_torch_A
        x_torch_seg_C = x_torch_seg @ self.C
        x_torch_seg_CB = x_torch_seg_C @ self.B  # M * [AC, AD, BC, BD], (M, 2, 4)
        sin_ACAD = x_torch_seg_CB[:, 0, 0] * x_torch_seg_CB[:, 1, 1] - x_torch_seg_CB[:, 1, 0] * x_torch_seg_CB[:, 0, 1]
        sin_BCBD = x_torch_seg_CB[:, 0, 2] * x_torch_seg_CB[:, 1, 3] - x_torch_seg_CB[:, 1, 2] * x_torch_seg_CB[:, 0, 3]
        sin_ACAD_BCBD = 0.5 * torch.relu(-sin_ACAD * sin_BCBD)**2
        sin_CACB = x_torch_seg_CB[:, 0, 0] * x_torch_seg_CB[:, 1, 2] - x_torch_seg_CB[:, 1, 0] * x_torch_seg_CB[:, 0, 2]
        sin_DADB = x_torch_seg_CB[:, 0, 1] * x_torch_seg_CB[:, 1, 3] - x_torch_seg_CB[:, 1, 1] * x_torch_seg_CB[:, 0, 3]
        sin_CACB_DADB = 0.5 * torch.relu(-sin_CACB * sin_DADB)**2
        sin_ACAD_BCBD_CACB_DADB = sin_ACAD_BCBD * sin_CACB_DADB
        res = sin_ACAD_BCBD_CACB_DADB.sum()
        return res.detach().numpy().reshape(-1)

    def grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        x_torch_A = x_torch.reshape(1, 2, -1) @ self.A
        x_torch_seg = x_torch_A
        x_torch_seg_C = x_torch_seg @ self.C
        x_torch_seg_CB = x_torch_seg_C @ self.B  # M * [AC, AD, BC, BD], (M, 2, 4)
        sin_ACAD = x_torch_seg_CB[:, 0, 0] * x_torch_seg_CB[:, 1, 1] - x_torch_seg_CB[:, 1, 0] * x_torch_seg_CB[:, 0, 1]
        sin_BCBD = x_torch_seg_CB[:, 0, 2] * x_torch_seg_CB[:, 1, 3] - x_torch_seg_CB[:, 1, 2] * x_torch_seg_CB[:, 0, 3]
        sin_ACAD_BCBD = 0.5 * torch.relu(-sin_ACAD * sin_BCBD)**2
        sin_CACB = x_torch_seg_CB[:, 0, 0] * x_torch_seg_CB[:, 1, 2] - x_torch_seg_CB[:, 1, 0] * x_torch_seg_CB[:, 0, 2]
        sin_DADB = x_torch_seg_CB[:, 0, 1] * x_torch_seg_CB[:, 1, 3] - x_torch_seg_CB[:, 1, 1] * x_torch_seg_CB[:, 0, 3]
        sin_CACB_DADB = 0.5 * torch.relu(-sin_CACB * sin_DADB)**2
        sin_ACAD_BCBD_CACB_DADB = sin_ACAD_BCBD * sin_CACB_DADB
        res = sin_ACAD_BCBD_CACB_DADB.sum()
        res.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy

    def thick_cons_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch_ = x_torch.reshape(2, -1)
        c1 = ((x_torch_[0] - CONFIG_BRACKET['lY'] / 2) ** 2 + (x_torch_[1] - CONFIG_BRACKET['zC']) ** 2)**0.5- (CONFIG_BRACKET['rC'] + 0.003)
        c2 = ((x_torch_[0] - CONFIG_BRACKET['bDia'] / 2) ** 2 + (x_torch_[1] - CONFIG_BRACKET['lZ'] + CONFIG_BRACKET['bDia']) ** 2)**0.5 - (CONFIG_BRACKET['bDia'] / 2 + 0.003)
        c3 = ((x_torch_[0] - CONFIG_BRACKET['lY'] / 2) ** 2 + (x_torch_[1] - CONFIG_BRACKET['lZ']) ** 2)**0.5 - (CONFIG_BRACKET['dCmp'] / 2 + 0.003)
        per_x_torch_ = x_torch_@self.Per
        c4 = ((x_torch_ - per_x_torch_)**2).sum(0)**0.5 - 0.003
        c1c2c3c4 = 0.5 * (torch.relu(-c1)**2 + torch.relu(-c2)**2 + torch.relu(-c3)**2 + torch.relu(-c4)**2)
        return c1c2c3c4.sum().detach().numpy().reshape(-1)

    def thick_con_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)
        x_torch_ = x_torch.reshape(2, -1)
        c1 = ((x_torch_[0] - CONFIG_BRACKET['lY'] / 2) ** 2 + (x_torch_[1] - CONFIG_BRACKET['zC']) ** 2) ** 0.5 - (CONFIG_BRACKET['rC'] + 0.003)
        c2 = ((x_torch_[0] - CONFIG_BRACKET['bDia'] / 2) ** 2 + (x_torch_[1] - CONFIG_BRACKET['lZ'] + CONFIG_BRACKET['bDia']) ** 2) ** 0.5 - (CONFIG_BRACKET['bDia'] / 2 + 0.003)
        c3 = ((x_torch_[0] - CONFIG_BRACKET['lY'] / 2) ** 2 + (x_torch_[1] - CONFIG_BRACKET['lZ']) ** 2) ** 0.5 - (CONFIG_BRACKET['dCmp'] / 2 + 0.003)
        per_x_torch_ = x_torch_ @ self.Per
        c4 = ((x_torch_ - per_x_torch_) ** 2).sum(0) ** 0.5 - 0.003
        c1c2c3c4 = 0.5 * (torch.relu(-c1) ** 2 + torch.relu(-c2) ** 2 + torch.relu(-c3) ** 2 + torch.relu(-c4) ** 2)
        y_torch = c1c2c3c4.sum()
        y_torch.backward()

        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy

    def convex_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch_A = x_torch.reshape(2, -1) @ self.A
        P01 = x_torch_A[:, 1:] - x_torch_A[:, :-1]
        dot = torch.sum(-P01[:, :-1] * P01[:, 1:], dim=0) - (P01[:, 0]*P01[:, -1]).sum()
        # cross = -(P01[0, :-1] * P01[1, 1:] - P01[1, :-1] * P01[0, 1:])
        flag = 0.5 * torch.relu(dot)**2 # *torch.relu(cross)
        return flag.sum().detach().numpy().reshape(-1)

    def convex_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        x_torch_A = x_torch.reshape(2, -1) @ self.A
        P01 = x_torch_A[:, :-1] - x_torch_A[:, 1:]
        dot = torch.sum(P01[:, :-1] * P01[:, 1:], dim=0) - (P01[:, 0]*P01[:, -1]).sum()
        # cross = -(P01[0, :-1] * P01[1, 1:] - P01[1, :-1] * P01[0, 1:])
        flag = 0.5 * torch.relu(dot)**2 # *torch.relu(cross)

        y_torch = flag.sum()
        y_torch.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy


class AENet:
    def __init__(self, ae, mu, sig):
        self.ae = ae
        self.mu = torch.from_numpy(mu).float()
        self.sig = torch.from_numpy(sig).float()

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def eval(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x.reshape(1, 2, -1))
        recon = self.ae(xn)
        return (torch.abs(xn - recon).mean()).detach().numpy().reshape(-1)

    def grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch.reshape(1, 2, -1))
        recon = self.ae(xn)
        y = torch.abs(xn - recon).mean()
        y.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy


class Areaer:
    def __init__(self, bs):
        super(Areaer, self).__init__()
        self.bs = bs
        self.gradient_steps = 0

    @staticmethod
    def relu(val):
        if val > 0:
            return val
        else:
            return np.zeros_like(val)

    def eval(self, x):
        # return -self.bs.eval(x)
        return 0.20968 - self.bs.eval(x) * 7850 * 2.0 * 0.004

    def grad(self, x):
        # return -self.bs.grad(x)
        self.gradient_steps = self.gradient_steps + 1
        return -self.bs.grad(x) * 7850 * 2.0 * 0.004

    def fair_eval(self, x):
        return 0.5 * self.relu(self.bs.fair_eval(x) - 3)**2

    def fair_grad(self, x):
        g = self.relu(self.bs.fair_eval(x) - 3) * self.bs.fair_grad(x)
        return g


class PhyNet:
    def __init__(self, net, mu, sig, area_scale):
        super(PhyNet, self).__init__()
        self.net = net
        self.mu = torch.from_numpy(mu).float()
        self.sig = torch.from_numpy(sig).float()
        self.area_scale = area_scale

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def freq_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        return self.net(self.normalized(x_torch.reshape(1, 2, -1)))[0].detach().numpy().reshape(-1)

    def vm_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        return self.net(self.normalized(x_torch.reshape(1, 2, -1)))[1].detach().numpy().reshape(-1)

    def area_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = self.net(self.normalized(x_torch.reshape(1, 2, -1)))[2].detach().numpy().reshape(-1)
        mass = 0.20968 - y_torch * self.area_scale * 7850 * 2.0 * 0.004
        # return -self.net(self.normalized(x_torch.reshape(1, 2, -1)))[2].detach().numpy().reshape(-1)
        return mass

    def freq_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(self.normalized(x_torch.reshape(1, 2, -1)))
        y_torch[0].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def vm_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(self.normalized(x_torch.reshape(1, 2, -1)))
        y_torch[1].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def area_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(self.normalized(x_torch.reshape(1, 2, -1)))
        # y_torch[2].backward()
        mass = 0.20968 - y_torch[2] * self.area_scale * 7850 * 2.0 * 0.004
        mass.backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy


class ShapeOptConPts:
    def __init__(self, phy, ae, area, freq_scale, vm_scale, area_scale):
        self.phy = phy
        self.ae = ae
        self.area = area
        self.freq_scale = freq_scale
        self.vm_scale = vm_scale
        self.area_scale = area_scale

        self.A = np.eye(32*2)
        lb = np.load('config/cpt_lb.npy')
        ub = np.load('config/cpt_ub.npy')

        thk = 0.003

        lbx_ = np.clip(lb[:32], thk, np.inf)
        ubx_ = np.clip(ub[:32], -np.inf, CONFIG_BRACKET['lY'] / 2 - thk / 2)
        lby_ = np.clip(lb[32:], thk, np.inf)
        uby_ = np.clip(ub[32:], -np.inf, CONFIG_BRACKET['lZ'] - thk)
        self.lb = np.hstack([lbx_, lby_])
        self.ub = np.hstack([ubx_, uby_])

        self.area_hist = []
        self.cl_hist = []
        self.cd_hist = []
        self.clcd_hist = []
        self.ae_hist = []
        self.Niter = 0
        self.check_cross = BracketRequire()
        # self.PENALTY = 0
        self.area_scale = 1  # 0.20968

    @staticmethod
    def mass(area):
        return area

    def mass_eval_bs(self, x):
        return self.area.eval(x) / self.area_scale

    def mass_grad_bs(self, x):
        return self.area.grad(x) / self.area_scale

    def mass_eval_phy(self, x):
        return self.phy.area_eval(x) / self.area_scale

    def mass_grad_phy(self, x):
        return self.phy.area_grad(x) / self.area_scale

    def print_init(self, xk):
        area_k = self.area.eval(xk)
        freq_k = self.phy.freq_eval(xk) * self.freq_scale
        vm_k = self.phy.vm_eval(xk) * self.vm_scale
        ae_k = self.ae.eval(xk)
        print('init area: ', self.mass(area_k),
              'init freq: ', freq_k,
              'init vm: ', vm_k,
              'init ae: ', ae_k)
        print('+++++++++++++++++++++++++++++++++++++++++')

    def minArea_conFreqVM(self, xdict):
        funcs = {}
        funcs['obj'] = self.mass_eval_bs(xdict['xvars']) + PENALTY * self.area.fair_eval(xdict['xvars'])
        funcs['con1'] = [
            self.A @ xdict['xvars'],
        ]
        funcs['con2'] = [
            self.phy.freq_eval(xdict['xvars']).item(),
            self.phy.vm_eval(xdict['xvars']).item(),
            self.ae.eval(xdict['xvars']).item(),
            # self.check_cross.eval(xdict['xvars']).item(),
            # self.check_cross.thick_cons_eval(xdict['xvars']).item(),
            # self.check_cross.convex_eval(xdict['xvars']).item()
        ]
        fail = False
        return funcs, fail

    def minArea_conFreqVM_sens(self, xdict, fdict):
        sens = {
            'obj': {'xvars': self.mass_grad_bs(xdict['xvars']) + PENALTY * self.area.fair_grad(xdict['xvars'])},
            'con2': {'xvars': np.vstack(
                [
                    self.phy.freq_grad(xdict['xvars']),
                    self.phy.vm_grad(xdict['xvars']),
                    self.ae.grad(xdict['xvars']),
                    # self.check_cross.grad(xdict['xvars']),
                    # self.check_cross.thick_con_grad(xdict['xvars']),
                    # self.check_cross.convex_grad(xdict['xvars']),
                ]
            )}
        }
        return sens

    def solve_minArea_conFreqVM(self, x0, freq_lb, vm_ub, i=None):
        self.print_init(x0)
        # self.print_init(np.load('optimized_results/normal_area_freq_60.00_vm_90.00_penalty_1.00_shareNet_notscale_beta/res_06_echo2W/opt_cpt.npy'))
        # input('...')
        ae_eps = AE_EPS  # ae_mu - 3 * ae_sig
        # self.area_scale = 1.

        optProb = Optimization('shape opt', self.minArea_conFreqVM)
        optProb.addVarGroup('xvars', 64, value=x0, lower=self.lb, upper=self.ub)
        # optProb.addConGroup('con1', 64, lower=self.lb, upper=self.ub, linear=True, jac={'xvars': self.A})
        # optProb.addConGroup('con2', 4,
        #                     lower=[freq_lb / self.freq_scale, None,                  None,    0],  # FREQ_LB/self.freq_scale
        #                     upper=[None,                      vm_ub / self.vm_scale, AE_EPS,  0])  # VM_UB/self.vm_scale
        optProb.addConGroup('con2', 3,
                            lower=[freq_lb / self.freq_scale, None,                None],  # FREQ_LB/self.freq_scale
                            upper=[None,                      vm_ub / self.vm_scale, AE_EPS])  # VM_UB/self.vm_scale

        optProb.addObj('obj')
        print(optProb)
        optOption = {'IPRINT': -1, 'MIT': MAX_ITER}
        opt = PSQP(options=optOption)
        # optOption = {'max_iter': MAX_ITER, 'tol': 1e-6}
        # opt = IPOPT(options=optOption)
        sol = opt(optProb, sens=self.minArea_conFreqVM_sens)
        print(sol)
        xs = np.array([v.value for v in sol.variables['xvars']])
        # input('press anything to continue...')

        freq0 = self.phy.freq_eval(x0) * self.freq_scale
        vm0 = self.phy.vm_eval(x0) * self.vm_scale
        freq1 = self.phy.freq_eval(xs) * self.freq_scale
        vm1 = self.phy.vm_eval(xs) * self.vm_scale
        out_str = '''
            mass: %f --> %f
            freq: %f --> %f
            vm  : %f --> %f
            ae  : %f --> %f
            cross: %f --> %f
            fair: %f --> %f
            sign: %f --> %f
        ''' % (self.mass(self.area.eval(x0)), self.mass(self.area.eval(xs)), freq0, freq1, vm0, vm1,
               self.ae.eval(x0), self.ae.eval(xs), self.check_cross.eval(x0), self.check_cross.eval(xs),
               self.area.bs.fair_eval(x0), self.area.bs.fair_eval(xs),
               self.check_cross.convex_eval(x0), self.check_cross.convex_eval(xs))
        print(out_str)
        print('gradients: ', self.area.gradient_steps)
        if i is None:
            self.area.bs.show(x0, filename=RESULT_DIR + '/init')
            self.area.bs.show(xs, filename=RESULT_DIR + '/opt')
        else:
            # if self.ae.eval(xs) > 0.008 or True:
            #     return
            res_dir = RESULT_DIR + 'res_%02d_echo2W' % i
            try:
                os.mkdir(res_dir)
            except OSError:
                pass
            self.area.bs.show(x0, filename=res_dir + '/init')
            self.area.bs.show(xs, filename=res_dir + '/opt', message=out_str)


if __name__ == '__main__':
    # seed = 0
    args = parser.parse_args()

    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    ind = args.item  # out 1, in 0
    area_train = np.load('config/area_train.npy')
    ind_train = area_train > 0
    area_train = area_train[ind_train]
    data = np.load('config/cpt_train.npy').transpose((0, 2, 1)).reshape(-1, 64)
    data = data[ind_train]
    labels = np.load('config/labels_train.npy')
    labels = labels[ind_train]
    x0 = data[ind]

    areaer = Areaer(bs=BsplineArea())

    net = ResNet1d18Fit()
    check = torch.load('mean-teacher-al/results-al-semi/net_best_teacher.tar', map_location=torch.device('cpu'))
    net.load_state_dict(check['teacher'])
    net.cpu()
    net.eval()
    freq_scales, vm_scales, area_scales = check['scales'].numpy()
    # input(area_scales)
    phy = PhyNet(net=net, mu=check['option']['mu_paras'], sig=check['option']['sig_paras'],
                 area_scale=area_scales)

    # ae_net = AEcoder(ndf=32, ngf=32, embed_dim=16)
    ae_net = AEcoder(ndfs=[32, 32, 64, 64], ngfs=[32, 32, 64, 64], embed_dim=24)
    check_ae = torch.load('shape-anomaly-detection/results_aeAE_ID=24/ae_best_zeros.tar', map_location=torch.device('cpu'))
    ae_net.load_state_dict(check_ae['net_state_dict'])
    ae_net.cpu()
    ae_net.eval()
    ae = AENet(ae=ae_net, mu=check_ae['option']['mu'], sig=check_ae['option']['sig'])
    opter = ShapeOptConPts(phy=phy, ae=ae, area=areaer,
                           freq_scale=freq_scales,
                           vm_scale=vm_scales,
                           area_scale=area_scales)

    opter.solve_minArea_conFreqVM(x0, freq_lb=FREQ_LB, vm_ub=VM_UB)


