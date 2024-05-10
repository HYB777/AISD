import os

import matplotlib.pyplot as plt
import torch
from bspline_utils import BsplineArea
from models import ResNet1d18Fit
from models_ae import AEcoder3
import numpy as np
from pyoptsparse import SLSQP, Optimization, PSQP
import scipy.io as scio
import random
import aerosandbox as asb
import argparse

random.seed(0)
np.random.seed(0)

THK = 0.01
MASS_SCALE = 6036.941971515103
AE_EPS = 0.02
MAX_ITER = 1000
VM_UB = 280. / 1.2
RESULT_DIR = 'optimized_results/ALSSL_vm_%.2f/' % VM_UB

parser = argparse.ArgumentParser()
parser.add_argument('--item', type=int, default=153775, help='data item number')

try:
    os.mkdir(RESULT_DIR)
except OSError:
    pass


class WingRibRequire:
    def __init__(self):
        super(WingRibRequire, self).__init__()
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

        Per = torch.eye(16)
        self.Per = torch.zeros((16, 16))
        self.Per[:8] = Per[8:]
        self.Per[8:] = Per[:8]

        self.asb = asb.Airfoil('naca4415')
        self.airfoil = torch.from_numpy(asb.Airfoil('naca4415').coordinates).float().reshape(1, -1, 2)
        self.airfoil_ub = torch.from_numpy(np.loadtxt('config/naca4415_upper.txt')).float().reshape(1, -1, 2)
        self.airfoil_lb = torch.from_numpy(np.loadtxt('config/naca4415_lower.txt')).float().reshape(1, -1, 2)

    @staticmethod
    def relu(x):
        return np.clip(x, 0, np.inf)

    def thick_cons_eval(self, x):
        x_torch = torch.tensor(x.reshape(3, 3, 2, 16), dtype=torch.float32)
        per_x_torch_ = x_torch.reshape(3, 3, 2, 16)@self.Per
        c1 = ((x_torch.reshape(3, 3, 2, 16) - per_x_torch_)**2).sum(2)**0.5 - THK
        x_torch_ = x_torch.reshape(9, 2, 16)
        x_torch_t = x_torch_.transpose(dim0=1, dim1=2)
        x_torch_t_ = x_torch_t.reshape(-1, 1, 2)
        c2 = ((x_torch_t_ - self.airfoil)**2).sum(-1)**0.5 - THK
        c1c2 = 0.5 * (torch.sum(torch.relu(-c1)**2) + torch.sum(torch.relu(-c2)**2))
        return c1c2.detach().numpy().reshape(-1)

    def thick_con_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        per_x_torch_ = x_torch.reshape(3, 3, 2, 16)@self.Per
        c1 = ((x_torch.reshape(3, 3, 2, 16) - per_x_torch_)**2).sum(2)**0.5 - THK
        x_torch_ = x_torch.reshape(9, 2, 16)
        x_torch_t = x_torch_.transpose(dim0=1, dim1=2)
        x_torch_t_ = x_torch_t.reshape(-1, 1, 2)
        c2 = ((x_torch_t_ - self.airfoil)**2).sum(-1)**0.5 - THK
        c1c2 = 0.5 * (torch.sum(torch.relu(-c1) ** 2) + torch.sum(torch.relu(-c2) ** 2))
        c1c2.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy

    def contain_eval(self, x):
        x = x.reshape(3, 3, 2, -1)
        thk = self.asb.local_thickness(x[:, :, 0])
        ub = self.asb.local_camber(x[:, :, 0]) + 0.5 * thk
        lb = self.asb.local_camber(x[:, :, 0]) - 0.5 * thk
        y = x[:, :, 1]
        y_ub = y - ub
        y_lb = y - lb
        y = self.relu(y_ub) + self.relu(-y_lb)
        return y.sum()

    def contain_grad(self, x):
        dx = 1 / 1000
        g = np.zeros_like(x)

        for i in range(len(x)):
            o = np.zeros_like(x)
            o[i] = dx / 2
            xp = x - o
            xq = x + o

            yp = self.contain_eval(xp)
            yq = self.contain_eval(xq)
            dy = yq - yp
            g[i] = dy / dx

        return g


class AENet:
    def __init__(self, ae, mu, sig):
        self.ae = ae
        self.mu = torch.from_numpy(mu).float()
        self.sig = torch.from_numpy(sig).float()

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def eval(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x.reshape(1, 3, 3, 2, -1))
        recon = self.ae(xn)
        return (torch.abs(xn - recon).mean()).detach().numpy().reshape(-1)

    def grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch.reshape(1, 3, 3, 2, -1))
        recon = self.ae(xn)
        y = torch.abs(xn - recon).mean()
        y.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy


class Areaer:
    def __init__(self, bs):
        super(Areaer, self).__init__()
        self.bs = bs

    def eval(self, x):
        return self.bs.eval(x) / MASS_SCALE

    def grad(self, x):
        return self.bs.grad(x) / MASS_SCALE


class PhyNet:
    def __init__(self, net, mu, sig):
        super(PhyNet, self).__init__()
        self.net = net
        self.mu = torch.from_numpy(mu).float()
        self.sig = torch.from_numpy(sig).float()

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def vm_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        return self.net(self.normalized(x_torch.reshape(1, 3, 3, 2, -1)))[0].detach().numpy().reshape(-1)

    def vm_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(self.normalized(x_torch.reshape(1, 3, 3, 2, -1)))
        y_torch[0].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy


class ShapeOptConPts:
    def __init__(self, phy, ae, area, vm_scale, mass_scale):
        self.phy = phy
        self.ae = ae
        self.area = area
        self.vm_scale = vm_scale
        self.mass_scale = mass_scale

        lb = np.load('config/cpt_lb.npy')
        ub = np.load('config/cpt_ub.npy')

        spars0_pts = np.loadtxt('config/spars0.txt')
        spars1_pts = np.loadtxt('config/spars1.txt')
        lb_ = np.ones_like(lb)
        ub_ = np.ones_like(ub)
        lb_[:, :, 0] = spars0_pts[0, 0] + THK
        lb_[:, :, 1] = -np.inf
        ub_[:, :, 0] = spars1_pts[-1, 0] - THK
        ub_[:, :, 1] = np.inf
        self.lb = np.clip(lb, lb_, ub_).reshape(-1)
        self.ub = np.clip(ub, lb_, ub_).reshape(-1)

        self.area_hist = []
        self.cl_hist = []
        self.cd_hist = []
        self.clcd_hist = []
        self.ae_hist = []
        self.Niter = 0

        self.check = WingRibRequire()

    def print_init(self, xk):
        area_k = self.area.eval(xk) * MASS_SCALE
        vm_k = self.phy.vm_eval(xk) * self.vm_scale
        ae_k = self.ae.eval(xk)
        contain_k = self.check.contain_eval(xk)
        thk_k = self.check.thick_cons_eval(xk)
        print('init area: ', area_k,
              'init vm: ', vm_k,
              'init ae: ', ae_k,
              'init thk: ', thk_k,
              'init contain: ', contain_k)
        print('+++++++++++++++++++++++++++++++++++++++++')

    def minArea_conFreqVM(self, xdict):
        funcs = {}
        funcs['obj'] = self.area.eval(xdict['xvars'])

        funcs['con1'] = [
            self.phy.vm_eval(xdict['xvars']).item(),
            self.ae.eval(xdict['xvars']).item(),
            self.check.thick_cons_eval(xdict['xvars']).item(),
        ]
        funcs['con2'] = [self.check.contain_eval(xdict['xvars']).item()]
        fail = False
        return funcs, fail

    def minArea_conFreqVM_sens(self, xdict, fdict):
        sens = {
            'obj': {'xvars': self.area.grad(xdict['xvars'])},
            'con1': {'xvars': np.vstack(
                [
                    self.phy.vm_grad(xdict['xvars']),
                    self.ae.grad(xdict['xvars']),
                    self.check.thick_con_grad(xdict['xvars']),
                ]
            )},
            'con2': {'xvars': self.check.contain_grad(xdict['xvars'])}
        }
        return sens

    def solve_minArea_conFreqVM(self, x0):
        self.print_init(x0)
        # self.print_init(np.load('optimized_results/vm_233.33/opt_cpt.npy'))
        # input('....')
        ae_eps = AE_EPS  # ae_mu - 3 * ae_sig

        optProb = Optimization('shape opt', self.minArea_conFreqVM)
        optProb.addVarGroup('xvars', 3*3*2*16, value=x0, lower=self.lb, upper=self.ub)
        optProb.addConGroup('con1', 3,
                            lower=[None,                  0,      0],  # FREQ_LB/self.freq_scale
                            upper=[VM_UB / self.vm_scale, ae_eps, 0])  # VM_UB/self.vm_scale
        optProb.addConGroup('con2', 1, lower=0, upper=0)  # VM_UB/self.vm_scale

        optProb.addObj('obj')
        print(optProb)
        optOption = {'IPRINT': -1, 'MIT': MAX_ITER}
        opt = PSQP(options=optOption)
        sol = opt(optProb, sens=self.minArea_conFreqVM_sens)
        print(sol)
        xs = np.array([v.value for v in sol.variables['xvars']])
        # input('press anything to continue...')

        vm0 = self.phy.vm_eval(x0) * self.vm_scale
        vm1 = self.phy.vm_eval(xs) * self.vm_scale
        out_str = '''
            mass: %f --> %f
            vm  : %f --> %f
            ae  : %f --> %f
        ''' % (self.area.eval(x0) * MASS_SCALE, self.area.eval(xs) * MASS_SCALE,
               vm0, vm1,
               self.ae.eval(x0), self.ae.eval(xs))
        print(out_str)
        self.area.bs.show(x0, filename=RESULT_DIR + '/init')
        self.area.bs.show(xs, filename=RESULT_DIR + '/opt')


if __name__ == '__main__':
    seed = 0
    args = parser.parse_args()

    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    ind = args.item
    data = np.load('data/holes_cpt_train.npy')
    labels = np.load('data/labels_train.npy')
    x0 = data[ind].reshape(-1)

    areaer = Areaer(bs=BsplineArea())

    net = ResNet1d18Fit()
    check = torch.load('mean-teacher-al/results-al-semi/net_best_teacher.tar', map_location=torch.device('cpu'))
    vm_scales, mass_scales = check['scales'].numpy()
    net.load_state_dict(check['teacher'])
    net.cpu()
    net.eval()
    phy = PhyNet(net=net, mu=check['option']['mu_paras'], sig=check['option']['sig_paras'])

    ae_net = AEcoder3(
        ndfs=[32, 32, 64, 64],
        ngfs=[32, 32, 64, 64],
        embed_dim=24,
    )
    check_ae = torch.load('learning-results-ae/results_ae_ID=24/ae_best_zeros.tar')
    ae_net.load_state_dict(check_ae['net_state_dict'])
    ae_net.cpu()
    ae_net.eval()
    ae = AENet(ae=ae_net, mu=check_ae['option']['mu'], sig=check_ae['option']['sig'])
    
    opter = ShapeOptConPts(phy=phy, ae=ae, area=areaer,
                           vm_scale=vm_scales,
                           mass_scale=mass_scales)

    opter.solve_minArea_conFreqVM(x0)
    # print(labels[ind])



