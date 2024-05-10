import os
import matplotlib.pyplot as plt
import torch
from copy import copy
from nets import *
from ae_nets import *
# from bem_utils.bs_utils import BsplineArea
# from bem_utils.rotor_calculator import param2sol
import bem_utils
from pyoptsparse import SLSQP, SNOPT, Optimization, PSQP, NLPQLP, IPOPT
import numpy as np
from functools import partial
import argparse


N_PARAMS = 5
RESULT_DIR = 'optimize_results/minCQ_conCT'
MAXITERS = 1000


parser = argparse.ArgumentParser()
parser.add_argument('--item', type=int, default=122848168, help='data item number')


class Areaer:
    def __init__(self):
        super(Areaer, self).__init__()
        self.bs = bem_utils.bs_utils.BsplineArea(34)

    def vol_eval(self, x):
        # 'radius': [0.61, 1.70, 4.77, 7.00],
        # 'chord': [0.57, 0.60, 0.50, 0.44],
        x_ = x[:-N_PARAMS]
        xn = x_.reshape(4, -1)
        sec_area = np.array([self.bs.eval(xn[i]) for i in range(4)])
        scales = np.array([0.57, 0.60, 0.50, 0.44]) ** 2
        si = sec_area * scales
        radiis = np.array([0.61, 1.70, 4.77, 7.00])
        h = radiis[1:] - radiis[:-1]  # [h12, h23, h34], [h12, h12+h23, h23+h34, h34]
        hij = np.array([h[0], h[0] + h[1], h[1] + h[2], h[2]]) / 2.
        return sum(si*hij)

    def vol_grad(self, x):
        # 'radius': [0.61, 1.70, 4.77, 7.00],
        # 'chord': [0.57, 0.60, 0.50, 0.44],
        sec_grad = np.array([self.grad(x, i).reshape(-1) for i in range(4)])
        scales = np.array([0.57, 0.60, 0.50, 0.44]) ** 2
        si = scales.reshape(4, -1) * sec_grad
        radiis = np.array([0.61, 1.70, 4.77, 7.00])
        h = radiis[1:] - radiis[:-1]  # [h12, h12+h23, h23+h34, h34]
        hij = np.array([h[0], h[0] + h[1], h[1] + h[2], h[2]]) / 2.
        return sum(si*hij.reshape(4, -1)).reshape(1, -1)

    def eval(self, x, i):
        x_ = x[:-N_PARAMS]
        xn = x_.reshape(4, -1)
        return self.bs.eval(xn[i])

    def grad(self, x, i):
        g = np.zeros_like(x)

        x_ = x[:-N_PARAMS]
        xn = x_.reshape(4, -1)
        gi = self.bs.grad(xn[i])
        g[i*64:(i+1)*64] = gi

        return g.reshape(1, -1)


class AENet:
    def __init__(self, ae, mu, sig):
        self.ae = ae
        self.mu = mu
        self.sig = sig

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def eval_(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x[:-N_PARAMS].reshape(4, 2, -1))
        recon = self.ae(xn)
        return (torch.abs(xn - recon).mean(dim=(1, 2))).detach().numpy().reshape(-1)

    def eval_i(self, x, i):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x[:-N_PARAMS].reshape(4, 1, 2, -1))
        recon = self.ae(xn[i])
        return (torch.abs(xn[i] - recon).mean()).detach().numpy().reshape(-1)

    def grad_i(self, x, i):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch[:-N_PARAMS].reshape(4, 1, 2, -1))
        recon = self.ae(xn[i])
        y = torch.abs(xn[i] - recon).mean()
        y.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy

    def eval(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x[:-N_PARAMS].reshape(4, 2, -1))
        recon = self.ae(xn)
        return (torch.abs(xn - recon).mean()).detach().numpy().reshape(-1)

    def grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch[:-N_PARAMS].reshape(4, 2, -1))
        recon = self.ae(xn)
        y = torch.abs(xn - recon).mean()
        y.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy


class CFDNet:
    def __init__(self, net, mu_paras, sig_paras):
        super(CFDNet, self).__init__()
        self.net = net
        self.mu_paras = torch.from_numpy(mu_paras).float()
        self.sig_paras = torch.from_numpy(sig_paras).float()
        self.gradient_steps = 0

    def ct_cq_eval(self, x, n):
        x_torch = torch.tensor(x, dtype=torch.float32)
        n_blade = torch.tensor(n - 2, dtype=torch.long).reshape(1)
        ct_cq = self.net(x_torch[:4*64].reshape(1, 4 * 2, -1),
                         n_blade,
                         (x_torch[4*64+1:].reshape(1, 4) - self.mu_paras) / self.sig_paras)
        return ct_cq.detach().numpy().reshape(-1)

    def eta_eval(self, x, n):
        ct_cq = self.ct_cq_eval(x, n)
        return ct_cq[0] / ct_cq[1]

    def ct_eval(self, x, n):
        return self.ct_cq_eval(x, n)[0]

    def cq_eval(self, x, n):
        return self.ct_cq_eval(x, n)[1]

    def ct_grad(self, x, n):
        x_torch = torch.tensor(x, dtype=torch.float32)
        n_blade = torch.tensor(n - 2, dtype=torch.long).reshape(1)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:4*64].reshape(1, 4 * 2, -1),
                           n_blade,
                           (x_torch[4*64+1:].reshape(1, 4) - self.mu_paras) / self.sig_paras).reshape(-1)
        y_torch[0].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def cq_grad(self, x, n):
        x_torch = torch.tensor(x, dtype=torch.float32)
        n_blade = torch.tensor(n - 2, dtype=torch.long).reshape(1)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:4*64].reshape(1, 4 * 2, -1),
                           n_blade,
                           (x_torch[4*64+1:].reshape(1, 4) - self.mu_paras) / self.sig_paras).reshape(-1)
        y_torch[1].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def eta_grad(self, x, n):
        x_torch = torch.tensor(x, dtype=torch.float32)
        n_blade = torch.tensor(n - 2, dtype=torch.long).reshape(1)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:4*64].reshape(1, 4 * 2, -1),
                           n_blade,
                           (x_torch[4*64+1:].reshape(1, 4) - self.mu_paras) / self.sig_paras).reshape(-1)
        eta = y_torch[0] / y_torch[1]
        eta.backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy


class ShapeOptConPts:
    def __init__(self, cfd, ae, ct_scale, cq_scale, J):
        self.cfd = cfd
        self.ae = ae
        self.arear = Areaer()

        self.ct_scale = ct_scale
        self.cq_scale = cq_scale
        self.J = J

        self.bs = bem_utils.bs_utils.BsplineArea(34)

        I = np.eye(32)
        O = np.zeros((32, 32))
        self.A = np.zeros((32 * 4, 64 * 4 + N_PARAMS))
        A_ = np.zeros((32 * 4, 64 * 4))
        A_[:32, :64] = np.hstack([I, -I])
        A_[32:32 * 2, 64:64 * 2] = np.hstack([I, -I])
        A_[32 * 2:32 * 3, 64 * 2:64 * 3] = np.hstack([I, -I])
        A_[32 * 3:, 64 * 3:] = np.hstack([I, -I])
        # A_[32 * 4:, 64 * 4:] = np.hstack([I, -I])
        self.A[:, :-N_PARAMS] = A_

        self.B = np.zeros((32 * 4, 64 * 4 + N_PARAMS))
        B_ = np.zeros((32 * 4, 64 * 4))
        B_[:32, :64] = np.hstack([I, O])
        B_[32:32 * 2, 64:64 * 2] = np.hstack([I, O])
        B_[32 * 2:32 * 3, 64 * 2:64 * 3] = np.hstack([I, O])
        B_[32 * 3:, 64 * 3:] = np.hstack([I, O])
        # B_[32 * 4:, 64 * 4:] = np.hstack([I, O])
        self.B[:, :-N_PARAMS] = B_

        self.c2 = np.zeros((4, 64 * 4 + N_PARAMS))
        self.c3 = np.zeros((N_PARAMS, 64 * 4 + N_PARAMS))

        self.c2[0, 32] = 1
        self.c2[1, 32 + 64] = 1
        self.c2[2, 32 + 128] = 1
        self.c2[3, 32 + 192] = 1
        # self.c2[4, 32 + 256] = 1

        self.c3[:, -N_PARAMS:] = np.eye(N_PARAMS)

        self.PARAMS_LB = [2, 20, 15, 5, -5]
        self.PARAMS_UB = [5, 25, 20, 10, 0]

        self.Niter = 0
        self.alfa = None

    def get_rot_mat(self, twist_angle):
        twist_angle = twist_angle * np.pi / 180
        cos_psi_ = np.cos(twist_angle)
        sin_psi_ = np.sin(twist_angle)
        rot = np.array([[cos_psi_, sin_psi_],
                        [-sin_psi_, cos_psi_]])
        return rot

    def print_paras(self, x0, xs):

        paras0 = x0[-N_PARAMS:]
        paras1 = xs[-N_PARAMS:]
        print(
            '\n#blades: ', paras0[0], ' --> ', paras1[0],
            '\npitches: ', paras0[1:], ' --> ', paras1[1:],
        )

    def solve(self, x, filename=None):

        a, a_down, a_up = self.bs.sampling(x[:64])
        b, b_down, b_up = self.bs.sampling(x[64:128])
        c, c_down, c_up = self.bs.sampling(x[128:192])
        d, d_down, d_up = self.bs.sampling(x[192:256])

        twist_a = self.get_rot_mat(x[-N_PARAMS:][-4])
        twist_b = self.get_rot_mat(x[-N_PARAMS:][-3])
        twist_c = self.get_rot_mat(x[-N_PARAMS:][-2])
        twist_d = self.get_rot_mat(x[-N_PARAMS:][-1])

        a_, a_down_, a_up_ = a @ twist_a.transpose(), a_down @ twist_a.transpose(), a_up @ twist_a.transpose()
        b_, b_down_, b_up_ = b @ twist_b.transpose(), b_down @ twist_b.transpose(), b_up @ twist_b.transpose()
        c_, c_down_, c_up_ = c @ twist_c.transpose(), c_down @ twist_c.transpose(), c_up @ twist_c.transpose()
        d_, d_down_, d_up_ = d @ twist_d.transpose(), d_down @ twist_d.transpose(), d_up @ twist_d.transpose()

        plt.close()
        plt.clf()
        fig, ax = plt.subplots(4, 1, sharex='col', sharey='row', figsize=(7, 12))
        ax[0].plot(a_down_[:, 0], a_down_[:, 1], 'ro-')
        ax[0].plot(a_up_[:, 0], a_up_[:, 1], 'ro-')
        ax[0].plot(a_[:, 0], a_[:, 1], 'g')
        ax[0].set_title('sec1')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_axis_off()
        ax[0].axis('equal')

        ax[1].plot(b_down_[:, 0], b_down_[:, 1], 'ro-')
        ax[1].plot(b_up_[:, 0], b_up_[:, 1], 'ro-')
        ax[1].plot(b_[:, 0], b_[:, 1], 'g')
        ax[1].set_title('sec2')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_axis_off()
        ax[1].axis('equal')

        ax[2].plot(c_down_[:, 0], c_down_[:, 1], 'ro-')
        ax[2].plot(c_up_[:, 0], c_up_[:, 1], 'ro-')
        ax[2].plot(c_[:, 0], c_[:, 1], 'g')
        ax[2].set_title('sec3')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_axis_off()
        ax[2].axis('equal')

        ax[3].plot(d_down_[:, 0], d_down_[:, 1], 'ro-')
        ax[3].plot(d_up_[:, 0], d_up_[:, 1], 'ro-')
        ax[3].plot(d_[:, 0], d_[:, 1], 'g')
        ax[3].set_title('sec4')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].set_axis_off()
        ax[3].axis('equal')

        plt.savefig(RESULT_DIR + '/' + filename + '_airfoils.svg')
        np.savetxt(RESULT_DIR + '/' + filename + '_sec1.dat', a)
        np.savetxt(RESULT_DIR + '/' + filename + '_sec2.dat', b)
        np.savetxt(RESULT_DIR + '/' + filename + '_sec3.dat', c)
        np.savetxt(RESULT_DIR + '/' + filename + '_sec4.dat', d)
        np.savetxt(RESULT_DIR + '/' + filename + '_sec1_.dat', a_)
        np.savetxt(RESULT_DIR + '/' + filename + '_sec2_.dat', b_)
        np.savetxt(RESULT_DIR + '/' + filename + '_sec3_.dat', c_)
        np.savetxt(RESULT_DIR + '/' + filename + '_sec4_.dat', d_)

        np.save(RESULT_DIR + '/' + filename + '.npy', x)

        self.solve_(x)

    def solve_(self, x):
        ct, cq, eta = bem_utils.rotor_calculator.param2sol(x)
        # print('ct=%.10f, cq=.10f, eta=.10f' % (ct, cq, eta))
        print('ct=', ct, ', cq=', cq, ', eta=', eta)

    def print_init(self, xk):
        # area_k = self.cfd.vol_eval(xk) * self.vol_scale
        ct_k = self.cfd.ct_eval(xk, int(xk[64*4])) * self.ct_scale
        cq_k = self.cfd.cq_eval(xk, int(xk[64*4])) * self.cq_scale
        ae_k = self.ae.eval(xk)
        print(
              'init ct: ', ct_k,
              'init cq: ', cq_k,
              'init eta: ', (ct_k / cq_k) * self.J,
              'init ae: ', ae_k)
        print('+++++++++++++++++++++++++++++++++++++++++')

    def minCQ_conCT(self, xdict, n_blade):
        funcs = {}
        funcs['obj'] = self.cfd.cq_eval(xdict['xvars'], n_blade)

        funcs['area'] = [
            # self.arear.eval(xdict['xvars'], 0),
            # self.arear.eval(xdict['xvars'], 1),
            # self.arear.eval(xdict['xvars'], 2),
            # self.arear.eval(xdict['xvars'], 3),
            self.arear.vol_eval(xdict['xvars'])
        ]

        funcs['con1'] = [
            self.A @ xdict['xvars'],
        ]
        funcs['con2'] = [
            self.cfd.ct_eval(xdict['xvars'], n_blade).item(),
            self.ae.eval_i(xdict['xvars'], 0).item(),
            self.ae.eval_i(xdict['xvars'], 1).item(),
            self.ae.eval_i(xdict['xvars'], 2).item(),
            self.ae.eval_i(xdict['xvars'], 3).item(),
            # self.ae.eval_i(xdict['xvars'], 4).item(),
        ]
        funcs['con3'] = [
            self.B @ xdict['xvars'],
        ]
        funcs['con4'] = [
            self.c2 @ xdict['xvars'],
        ]
        funcs['con5'] = [
            self.c3 @ xdict['xvars'],
        ]
        fail = False
        return funcs, fail

    def minCQ_conCT_sens(self, xdict, fdict, n_blade):
        sens = {
            'obj': {'xvars': self.cfd.cq_grad(xdict['xvars'], n_blade)},

            # 'area': {'xvars': np.vstack(
            #     [
            #         self.arear.grad(xdict['xvars'], 0),
            #         self.arear.grad(xdict['xvars'], 1),
            #         self.arear.grad(xdict['xvars'], 2),
            #         self.arear.grad(xdict['xvars'], 3),
            #     ]
            # )},
            'area': {'xvars': self.arear.vol_grad(xdict['xvars'])},

            'con2': {'xvars': np.vstack(
                [
                    self.cfd.ct_grad(xdict['xvars'], n_blade),
                    self.ae.grad_i(xdict['xvars'], 0),
                    self.ae.grad_i(xdict['xvars'], 1),
                    self.ae.grad_i(xdict['xvars'], 2),
                    self.ae.grad_i(xdict['xvars'], 3),
                    # self.ae.grad_i(xdict['xvars'], 4),
                ]
            )},
        }
        return sens

    def solve_minCQ_conCT_(self, x0, ct0, n_blade, areas_lb):

        ae_eps = 1e-3  # 8e-4, ae_mu - 3 * ae_sig

        thick = self.A @ x0

        # minCQ_conCT = partial(self.minCQ_conCT, n_blade=n_blade)
        # minCQ_conCT_sens = partial(self.minCQ_conCT_sens, n_blade=n_blade)

        minCQ_conCT = lambda xdict: self.minCQ_conCT(xdict, n_blade=n_blade)
        minCQ_conCT_sens = lambda xdict, fdict: self.minCQ_conCT_sens(xdict, fdict, n_blade=n_blade)

        optProb = Optimization('shape opt', minCQ_conCT)
        optProb.addVarGroup('xvars', 4 * 64 + N_PARAMS, value=x0)
        optProb.addConGroup('con1', 32 * 4, lower=thick * 0.0, linear=True, jac={'xvars': self.A})
        optProb.addConGroup('con2', 5, lower=[ct0 / self.ct_scale] + [None] * 4, upper=[None] + [ae_eps] * 4)
        # optProb.addConGroup('con2', 4, lower=[None] * 4, upper=[ae_eps] * 4)
        optProb.addConGroup('area', 1, lower=areas_lb * 0.1, upper=None)
        optProb.addConGroup('con3', 32 * 4, lower=np.zeros(32 * 4), linear=True, jac={'xvars': self.B})
        optProb.addConGroup('con4', 4, upper=0, linear=True, jac={'xvars': self.c2})
        optProb.addConGroup('con5', N_PARAMS, lower=self.PARAMS_LB, upper=self.PARAMS_UB, linear=True,
                            jac={'xvars': self.c3})
        # optProb.addConGroup('aspect', 1, lower=None, upper=aspect0.item())
        optProb.addObj('obj')
        # print(optProb)
        optOption = {'IPRINT': -1, 'MIT': MAXITERS}
        opt = PSQP(options=optOption)
        # optOption = {'max_iter': 1000}
        # opt = IPOPT(options=optOption)

        sol = opt(optProb, sens=minCQ_conCT_sens)
        # sol = opt(optProb, sens='FD')
        # print(sol)
        xs = np.array([v.value for v in sol.variables['xvars']])
        # input('press anything to continue...')

        print('==============================#blade=%d=======================================' % n_blade)
        ct1 = self.cfd.ct_eval(xs, n_blade) * self.ct_scale
        cq1 = self.cfd.cq_eval(xs, n_blade) * self.cq_scale
        eta1 = (ct1 / cq1) * self.J
        return xs, ct1, cq1, eta1

    def solve_minCQ_conCT(self, x0):

        self.print_init(x0)
        ct0 = self.cfd.ct_eval(x0, int(x0[64 * 4])).item() * self.ct_scale
        cq0 = self.cfd.cq_eval(x0, int(x0[64 * 4])).item() * self.cq_scale

        areas = [self.arear.eval(x0, i) for i in range(4)]
        areas_lb = [a * 0.1 for a in areas]
        vol_lb = self.arear.vol_eval(x0)
        print('areas = ', areas, 'vol = ', vol_lb)

        n_blades = [2, 3, 4, 5]

        sols = [self.solve_minCQ_conCT_(x0, ct0, n_blade_, vol_lb) for n_blade_ in n_blades]

        input('press anything to continue...')

        print('=====================================================================')
        max_eta = -1  # (ct0 / cq0) * self.J
        idx = -1
        xs = copy(x0)
        for i in range(4):
            soli = sols[i]
            print('#blades=%d: ct=%.10f, cq=%.10f, eta=%.10f' % (n_blades[i], soli[1], soli[2], soli[3]))
            if soli[3] > max_eta:
                idx = i
                max_eta = soli[3]
                xs = copy(soli[0])

        xs[64*4] = n_blades[idx]
        self.print_paras(x0, xs)
        self.print_init(x0)
        self.print_init(xs)
        # print(x0[64*4])
        # input('wait ...')
        self.solve(x0, 'init')
        self.solve(xs, 'opt')
        return xs

    def compare(self):
        # x_init = np.load('optimize_results/minCQ_conCT/init.npy')
        # x_opt = np.load('optimize_results/minCQ_conCT/opt.npy')

        for s in ['init', 'opt']:
            x0 = np.load('optimize_results/minCQ_conCT/%s.npy' % s)
            ct0 = self.cfd.ct_eval(x0, int(x0[64 * 4])).item() * self.ct_scale
            cq0 = self.cfd.cq_eval(x0, int(x0[64 * 4])).item() * self.cq_scale
            print(s, ct0, cq0)


if __name__ == '__main__':
    args = parser.parse_args()
    try:
        os.mkdir(RESULT_DIR)
    except FileExistsError:
        pass

    base2 = 64*4
    pitches = np.load('./rotors_dataset/pitches.npy')
    airfoils = np.load('./airfoils/conts_wgan.npy')
    perms = np.load('./rotors_dataset/rotor_keys.npy')
    res = np.load('./rotors_dataset/rotor_values.npy')
    n = len(airfoils)
    airfoil = airfoils.reshape((n, 2, 32))
    res = res.reshape((-1, 10))
    n_blade = res[:, 0]
    ct = res[:, -5]
    cq = res[:, -4]
    eta = res[:, -2]
    J = res[:, -1]

    ind = args.item

    perms = perms[ind // base2]
    data = airfoils[perms]
    data = data.reshape((-1, 32))
    pitch = pitches[ind % 64]
    n_blade = n_blade[ind]
    ct = ct[ind]
    cq = cq[ind]
    eta = eta[ind]
    J = J[ind]
    print(ct, cq, eta, J, n_blade)
    x0 = np.hstack([data.reshape(-1), n_blade, pitch])

    net = ResNet1d18_Rotor()

    def _init_fn(worker_id):
        np.random.seed(1 + worker_id)


    check = torch.load('results_al/net_best_teacher.tar', map_location=torch.device('cpu'))
    mu_paras = check['option']['mu_paras']
    sig_paras = check['option']['sig_paras']
    net.load_state_dict(check['teacher'])
    net.cpu()
    net.eval()
    cfd = CFDNet(net=net, mu_paras=mu_paras, sig_paras=sig_paras)

    df = 4
    ae_net = AEcoder(ndf=df, ngf=df)
    check_ae = torch.load('../airfoil_design/shape-anomaly-detection/resultsAE_ID=16_wgan/ae_best_zeros.tar',
                          map_location=torch.device('cpu'))
    ae_net.load_state_dict(check_ae['net_state_dict'])
    ae_net.cpu()
    ae_net.eval()
    ae = AENet(ae=ae_net, mu=check_ae['option']['mu'], sig=check_ae['option']['sig'])

    scales = check['scales']
    scale_ct = scales[0].item()
    scale_cq = scales[1].item()

    opter = ShapeOptConPts(cfd=cfd, ae=ae, ct_scale=scale_ct, cq_scale=scale_cq, J=0.17349727/(2*np.pi))
    xs = opter.solve_minCQ_conCT(x0)
    opter.compare()



