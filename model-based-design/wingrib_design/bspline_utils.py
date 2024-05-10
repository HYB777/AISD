import numpy as np
import math
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import aerosandbox as asb
import scipy.io as scio
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

INDEX = 0
CONFIG = 'config'
LEN_T = 51


def _basis_(i, d, x, knots):

    if d == 0 :
        return ((knots[i] <= x < knots[i + 1]) or (x == 1 and knots[i + 1] == 1.0)) * 1.0
    val = 0
    if knots[i + d] != knots[i]:
        val += ((x - knots[i]) / (knots[i + d] - knots[i])) * _basis_(i, d - 1, x, knots)
    if knots[i + d + 1] != knots[i + 1]:
        val += ((knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1])) * _basis_(i + 1, d - 1, x, knots)
    return val


def _B(i, m, x, t):
    return (m / (t[i + m] - t[i])) * _basis_(i, m-1, x, t)


def T(i, k, x, j, l, y):
    if y[j] == y[j + l]:
        return _B(i, k, y[j], x) * math.factorial(k + l - 1) / (math.factorial(k) * math.factorial(l))
    elif x[i] == x[i + k]:
        return _B(j, l, x[i], y) * math.factorial(k + l - 1) / (math.factorial(k) * math.factorial(l))
    elif x[i + k] <= y[j + l] and y[j] <= y[j + l]:
        return ((x[i + k] - y[j]) * T(i, k, x, j, l - 1, y) +
                (y[j + l] - x[i + k]) * T(i, k, x, j + 1, l - 1, y)) / (y[j + l] - y[j]) + T(i, k - 1, x, j, l, y)
    elif y[j] <= x[i] and y[j] <= y[j + l]:
        return ((x[i] - y[j]) * T(i, k, x, j, l - 1, y) +
                (y[j + l] - x[i]) * T(i, k, x, j + 1, l - 1, y)) / (y[j + l] - y[j]) + T(i + 1, k - 1, x, j, l, y)
    elif x[i] <= y[j] and x[i] <= x[i + k]:
        return ((y[j] - x[i]) * T(i, k - 1, x, j, l, y) +
                (x[i + k] - y[j]) * T(i + 1, k - 1, x, j, l, y)) / (x[i + k] - x[i]) + T(i, k, x, j + 1, l - 1, y)
    elif y[j + l] <= x[i + k] and x[i] <= x[i + k]:
        return ((y[j + l] - x[i]) * T(i, k - 1, x, j, l, y) +
                (x[i + k] - y[j + l]) * T(i + 1, k - 1, x, j, l, y)) / (x[i + k] - x[i]) + T(i, k, x, j, l - 1, y)
    else:
        raise ModuleNotFoundError


def rib2hole_id(i):
    if i < 4:
        return 1
    elif i < 7:
        return 2
    else:
        return 3


def hole2rib_id(i):
    if i == 0:
        return [0, 1, 2, 3]
    elif i == 1:
        return [4, 5, 6]
    else:
        return [7, 8, 9]


class BsplineArea:
    def __init__(self, n=18, d=3):

        '''
        :param n: the number of control points is n + 1
        :param d: degree
        '''

        super(BsplineArea, self).__init__()
        airfoil_area = asb.Airfoil('naca4415').area()

        self.airfoil_u = np.loadtxt(CONFIG + '/naca4415_upper.txt')
        self.airfoil_l = np.loadtxt(CONFIG + '/naca4415_lower.txt')
        self.spars0_pts = np.loadtxt(CONFIG + '/spars0.txt')
        self.spars1_pts = np.loadtxt(CONFIG + '/spars1.txt')

        a1 = (self.spars0_pts[0] - self.spars0_pts[-1])[0] * (self.spars0_pts[0] - self.spars0_pts[1])[1]
        a2 = (self.spars0_pts[2] - self.spars0_pts[9])[0] * (self.spars0_pts[2] - self.spars0_pts[3])[1]
        b1 = (self.spars1_pts[0] - self.spars1_pts[-1])[0] * (self.spars1_pts[0] - self.spars1_pts[1])[1]
        b2 = (self.spars1_pts[2] - self.spars1_pts[9])[0] * (self.spars1_pts[2] - self.spars1_pts[3])[1]
        H_area = a2 + 2*a1 + b2 + 2*b1
        self.ref_area = airfoil_area - H_area

        self.gaussian_weight = np.array([(322 - 13 * 70**0.5) / 900,
                                         (322 + 13 * 70**0.5) / 900,
                                         128/225,
                                         (322 + 13 * 70**0.5) / 900,
                                         (322 - 13 * 70**0.5) / 900])

        self.gaussian_points = np.array([-(5 + 2 * (10 / 7)**0.5)**0.5/3,
                                         -(5 - 2 * (10 / 7)**0.5)**0.5/3,
                                         0,
                                         (5 - 2 * (10 / 7)**0.5)**0.5/3,
                                         (5 + 2 * (10 / 7)**0.5)**0.5/3])

        self.d = d
        self.dim = n
        self.dimOut = 1
        self.n = n
        self.n_control = n + 1
        np_float = np.float32
        self.N = np.zeros((LEN_T, self.n_control), dtype=np_float)
        self.NN = np.zeros((self.n_control, self.n_control), dtype=np_float)
        self.M = np.zeros((self.n_control, self.n_control - 1))
        self.T = np.zeros((self.n_control - 1, self.n_control))
        self.T2 = np.zeros((self.n_control - 2, self.n_control))
        self.MT = np.zeros((self.n_control, self.n_control), dtype=np_float)
        self.F = np.zeros((self.n_control - 2, self.n_control - 2))
        self.t = np.linspace(0, 1, (n + d + 1) + 1)
        self.Warp = np.zeros((self.n_control, self.n_control - 3))
        self.Warp[:self.n_control - 3] = np.eye(self.n_control - 3)
        self.Warp[self.n_control - 3:, :3] = np.eye(3)

        self.rho = 2770
        self.thk = 0.1
        chord_root = 7.88
        chord_tip = 1.25
        self.chords = []
        for i in range(1, 11):
            wi_ct = i / 10
            self.chords.append((1 - wi_ct) * chord_root + wi_ct * chord_tip)

        try:
            self.MT = np.load(CONFIG + '/MT.npy')
            self.M = np.load(CONFIG + '/M.npy')
            self.T = np.load(CONFIG + '/T.npy')
            self.T2 = np.load(CONFIG + '/T2.npy')
            self.F = np.load(CONFIG + '/F.npy')
        except FileNotFoundError:
            self.init_MT()

        try:
            self.N = np.load(CONFIG + '/N_u%d.npy' % LEN_T)
        except FileNotFoundError:
            self.init_N()

        self.WtMTMTtW = self.Warp.transpose() @ (self.MT - self.MT.transpose()) @ self.Warp
        self.WtT2tFT2W = self.Warp.transpose() @ self.T2.transpose() @ self.F @ self.T2 @ self.Warp

    def init_MT(self):
        M = np.zeros((self.n_control, self.n_control - 1))
        for i in range(self.n_control):
            for j in range(self.n_control - 1):
                tlb = max(self.t[i], self.t[j + 1])
                tub = min(self.t[i + self.d + 1], self.t[j + self.d + 1])
                if tub <= tlb:
                    continue
                if self.t[self.d] <= tlb and tub <= self.t[self.n+1]:
                    factor = ((self.t[i + self.d + 1] - self.t[i]) / (self.d + 1)) * ((self.t[j + self.d + 1] - self.t[j + 1]) / self.d)
                    M[i, j] = factor * T(i, self.d + 1, self.t, j + 1, self.d, self.t) * \
                              math.factorial(self.d + 1) * math.factorial(self.d) / math.factorial(self.d + self.d)
                else:
                    a = max(self.t[self.d], tlb)
                    b = min(self.t[self.n+1], tub)
                    if b > a:
                        txi = lambda xi: (b - a) / 2 * xi + (b + a) / 2
                        M[i, j] = (np.array([_basis_(i, self.d, ti, self.t)*_basis_(j + 1, self.d-1, ti, self.t)
                                             for ti in txi(self.gaussian_points)]) * self.gaussian_weight).sum() * (b - a) / 2

        print(M)
        self.M = M

        F = np.zeros((self.n_control - 2, self.n_control - 2))
        for i in range(self.n_control - 2):
            for j in range(self.n_control - 2):
                tlb = max(self.t[i + 2], self.t[j + 2])
                tub = min(self.t[i + self.d + 1], self.t[j + self.d + 1])
                if tub <= tlb:
                    continue
                if self.t[self.d] <= tlb and tub <= self.t[self.n+1]:
                    factor = ((self.t[i + self.d + 1] - self.t[i + 2]) / (self.d - 1)) * ((self.t[j + self.d + 1] - self.t[j + 2]) / (self.d - 1))
                    F[i, j] = factor * T(i + 2, self.d - 1, self.t, j + 2, self.d - 1, self.t) * \
                              math.factorial(self.d - 1) * math.factorial(self.d - 1) / math.factorial(self.d - 1 + self.d - 1 - 1)
                else:
                    a = max(self.t[self.d], tlb)
                    b = min(self.t[self.n+1], tub)
                    if b > a:
                        txi = lambda xi: (b - a) / 2 * xi + (b + a) / 2
                        F[i, j] = (np.array([_basis_(i + 2, self.d - 2, ti, self.t)*_basis_(j + 2, self.d - 2, ti, self.t)
                                             for ti in txi(self.gaussian_points)]) * self.gaussian_weight).sum() * (b - a) / 2

        self.F = F
        T_ = np.zeros((self.n_control - 1, self.n_control))
        T_[:, :-1] -= np.eye(self.n_control - 1)
        T_[:, 1:] += np.eye(self.n_control - 1)
        T_ = np.diag(self.d / (self.t[self.d+1:-1]-self.t[1:self.n + 1]))@T_
        self.T = T_
        self.MT = M@T_

        T_2 = np.zeros((self.n_control - 2, self.n_control - 1))
        T_2[:, :-1] -= np.eye(self.n_control - 2)
        T_2[:, 1:] += np.eye(self.n_control - 2)
        T_2 = np.diag((self.d - 1) / (self.t[self.d:-3]-self.t[1:self.n]))@T_2
        self.T2 = T_2 @ self.T

        np.save(CONFIG + '/M.npy', self.M)
        np.save(CONFIG + '/MT.npy', self.MT)
        np.save(CONFIG + '/T.npy', self.T)
        np.save(CONFIG + '/T2.npy', self.T2)
        np.save(CONFIG + '/F.npy', self.F)

    def init_N(self):
        u = np.linspace(self.t[self.d], self.t[self.n+1], LEN_T)
        for i in tqdm(range(len(u))):
            for j in range(self.n_control):
                self.N[i, j] = _basis_(j, self.d, u[i], self.t)
        np.save(CONFIG + '/N_u%d.npy' % LEN_T, self.N)

    def show(self, x, filename=None, message=None):
        # (3, 3, 2, 16)
        x = x.reshape(3, 3, 2, 16)
        c = np.zeros((3, 3, 2, LEN_T))

        titles = ['group 1: No.1 - No.4', 'group 2: No.5 - No.7', 'group 3: No.8 - No.10']

        for i in range(1):
            P0 = x[i, 0].transpose()

            P0c = P0.mean(0)

            w = 1.2
            FONTSIZE = 20
            eps = 0.015
            EPS = np.array([[eps*2, eps]])

            P0e = w * P0 + (1 - w) * (P0c[np.newaxis] + EPS)
            w *= 1.2
            P0e_ = w * P0 + (1 - w) * (P0c[np.newaxis] + EPS)

            P0 = np.vstack([P0, P0[:self.d]])

            plt.plot(P0[:, 0], P0[:, 1], 'ro-', label='Bspline control points')
            c0 = self.N @ P0

            c[i, 0] = c0.transpose()

            plt.plot(c0[:, 0], c0[:, 1], 'g-', label='curve')

            for k in range(3):
                plt.text(P0[k, 0], P0[k, 1], '$p^{i,j}_{%d}$($p^{i,j}_{%d}$)' % (k, k + 16),
                           fontsize=FONTSIZE, c='k')

            for k in range(len(P0e)):
                # print('$p^{%d, 1}_%d$' % (i+1, k))
                if k < 3:
                    continue
                plt.text(P0[k, 0], P0[k, 1], '$p^{i,j}_{%d}$' % (k), fontsize=FONTSIZE, c='k')

        plt.axis('equal')
        plt.show()
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        for i in range(3):
            P0 = x[i, 0].transpose()
            P1 = x[i, 1].transpose()
            P2 = x[i, 2].transpose()

            P0c = P0.mean(0)
            P1c = P1.mean(0)
            P2c = P2.mean(0)
            w = 1.2
            FONTSIZE = 15
            eps = 0.015
            EPS = np.array([[eps*2, eps]])

            P0e = w * P0 + (1 - w) * (P0c[np.newaxis] + EPS)
            P1e = w * P1 + (1 - w) * (P1c[np.newaxis] + EPS)
            P2e = w * P2 + (1 - w) * (P2c[np.newaxis] + EPS)
            w *= 1.2
            P0e_ = w * P0 + (1 - w) * (P0c[np.newaxis] + EPS)
            P1e_ = w * P1 + (1 - w) * (P1c[np.newaxis] + EPS)
            P2e_ = w * P2 + (1 - w) * (P2c[np.newaxis] + EPS)

            P0 = np.vstack([P0, P0[:self.d]])
            P1 = np.vstack([P1, P1[:self.d]])
            P2 = np.vstack([P2, P2[:self.d]])

            ax[i].plot(P0[:, 0], P0[:, 1], 'ro-', label='Bspline control points')
            ax[i].plot(P1[:, 0], P1[:, 1], 'ro-', label='Bspline control points')
            ax[i].plot(P2[:, 0], P2[:, 1], 'ro-', label='Bspline control points')

            # for k in range(3):
            #     ax[i].text(P0e_[k, 0], P0e_[k, 1], '($p^{%d,1}_{%d}$)' % (i + 1, k + 16),
            #                fontsize=FONTSIZE, c='b')
            #     ax[i].text(P1e_[k, 0], P1e_[k, 1], '($p^{%d,2}_{%d}$)' % (i + 1, k + 16),
            #                fontsize=FONTSIZE, c='b')
            #     ax[i].text(P2e_[k, 0], P2e_[k, 1], '($p^{%d,3}_{%d}$)' % (i + 1, k + 16),
            #                fontsize=FONTSIZE, c='b')
            #
            # for k in range(len(P0e)):
            #     # print('$p^{%d, 1}_%d$' % (i+1, k))
            #     ax[i].text(P0e[k, 0], P0e[k, 1], '$p^{%d,1}_{%d}$' % (i + 1, k), fontsize=FONTSIZE, c='b')
            #     ax[i].text(P1e[k, 0], P1e[k, 1], '$p^{%d,2}_{%d}$' % (i + 1, k), fontsize=FONTSIZE, c='b')
            #     ax[i].text(P2e[k, 0], P2e[k, 1], '$p^{%d,3}_{%d}$' % (i + 1, k), fontsize=FONTSIZE, c='b')
            #
            #     # if k < 3:
            #     #     ax[i].text(P0e[k, 0], P0e[k, 1], '$p^{%d,1}_{%d}$($p^{%d,1}_{%d}$)' % (i+1, k, i+1, k+16), fontsize=FONTSIZE)
            #     #     ax[i].text(P1e[k, 0], P1e[k, 1], '$p^{%d,2}_{%d}$($p^{%d,2}_{%d}$)' % (i+1, k, i+1, k+16), fontsize=FONTSIZE)
            #     #     ax[i].text(P2e[k, 0], P2e[k, 1], '$p^{%d,3}_{%d}$($p^{%d,3}_{%d}$)' % (i+1, k, i+1, k+16), fontsize=FONTSIZE)
            #     # else:
            #     #     print(k, len(P0) - 3)
            #     #     ax[i].text(P0e[k, 0], P0e[k, 1], '$p^{%d,1}_{%d}$' % (i+1, k), fontsize=FONTSIZE)
            #     #     ax[i].text(P1e[k, 0], P1e[k, 1], '$p^{%d,2}_{%d}$' % (i+1, k), fontsize=FONTSIZE)
            #     #     ax[i].text(P2e[k, 0], P2e[k, 1], '$p^{%d,3}_{%d}$' % (i+1, k), fontsize=FONTSIZE)

            c0 = self.N @ P0
            c1 = self.N @ P1
            c2 = self.N @ P2

            c[i, 0] = c0.transpose()
            c[i, 1] = c1.transpose()
            c[i, 2] = c2.transpose()

            ax[i].plot(c0[:, 0], c0[:, 1], 'g-', label='curve')
            ax[i].plot(c1[:, 0], c1[:, 1], 'g-', label='curve')
            ax[i].plot(c2[:, 0], c2[:, 1], 'g-', label='curve')

            ax[i].plot(self.airfoil_l[:, 0], self.airfoil_l[:, 1], 'k-')
            ax[i].plot(self.airfoil_u[:, 0], self.airfoil_u[:, 1], 'k-')
            ax[i].fill(self.spars0_pts[:, 0], self.spars0_pts[:, 1], 'b')
            ax[i].fill(self.spars1_pts[:, 0], self.spars1_pts[:, 1], 'b')

            ax[i].set_title(titles[i], fontsize=FONTSIZE*1.2)
            if i < 2:
                ax[i].set_xticks([])
                ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            # ax[i].set_yticks([])
            # ax[i].set_axis_off()

            ax[i].axis('equal')

        if filename is None:
            plt.show()
        else:
            plt.savefig('%s.svg' % filename)
            plt.clf()
            np.save('%s_cpt.npy' % filename, x.reshape(-1))
            np.save('%s_curve.npy' % filename, c)
            if message is None:
                scio.savemat('%s.mat' % filename, {'curve': c})
            else:
                scio.savemat('%s.mat' % filename, {'curve': c, 'message': message})

    def get_S(self, x):
        P = self.Warp @ (x.reshape(2, -1).transpose())
        Q = self.T @ P
        S = np.zeros((len(P), len(Q)))
        for i in range(len(P)):
            for j in range(len(Q)):
                S[i, j] = np.sign(P[i, 0] * Q[j, 1] - P[i, 1] * Q[j, 0])
        return S

    def eval(self, x):

        x = x.reshape(3, 3, 2, 16)
        mass = 0

        for i in range(3):
            P0 = x[i, 0]
            P1 = x[i, 1]
            P2 = x[i, 2]

            area_0 = -0.5 * P0[0] @ self.WtMTMTtW @ P0[1]
            area_1 = -0.5 * P1[0] @ self.WtMTMTtW @ P1[1]
            area_2 = -0.5 * P2[0] @ self.WtMTMTtW @ P2[1]
            area_012 = area_0 + area_1 + area_2
            factor_i = sum([self.chords[j] ** 2 for j in hole2rib_id(i)]) * self.thk * self.rho
            mass += (self.ref_area - area_012) * factor_i

        return mass

    def grad(self, x):
        x = x.reshape(3, 3, 2, 16)
        g = np.zeros_like(x)

        for i in range(3):
            P0 = x[i, 0]
            P1 = x[i, 1]
            P2 = x[i, 2]

            factor_i = sum([self.chords[j] ** 2 for j in hole2rib_id(i)]) * self.thk * self.rho
            dmassi_darea012 = -factor_i
            darea0_dP0 = -0.5 * self.WtMTMTtW @ P0[1], -0.5 * P0[0] @ self.WtMTMTtW
            darea1_dP1 = -0.5 * self.WtMTMTtW @ P1[1], -0.5 * P1[0] @ self.WtMTMTtW
            darea2_dP2 = -0.5 * self.WtMTMTtW @ P2[1], -0.5 * P2[0] @ self.WtMTMTtW
            g[i, 0, 0] = dmassi_darea012 * darea0_dP0[0]
            g[i, 0, 1] = dmassi_darea012 * darea0_dP0[1]

            g[i, 1, 0] = dmassi_darea012 * darea1_dP1[0]
            g[i, 1, 1] = dmassi_darea012 * darea1_dP1[1]

            g[i, 2, 0] = dmassi_darea012 * darea2_dP2[0]
            g[i, 2, 1] = dmassi_darea012 * darea2_dP2[1]

        return g.reshape(1, -1)

    def fair_eval(self, x):

        Px = x[:32]
        Py = x[32:]

        return Px @ self.WtT2tFT2W @ Px + Py @ self.WtT2tFT2W @ Py

    def fair_grad(self, x):
        Px = x[:32]
        Py = x[32:]

        gx = 2 * self.WtMTMTtW @ Px
        gy = 2 * self.WtMTMTtW @ Py

        return np.hstack([gx, gy])


if __name__ == '__main__':
    res = 'init'
    data = np.load('optimized_results/vm_233.33/%s_cpt.npy' % res).reshape(-1)
    # label = np.load('data/labels_train.npy')[ind]
    bs_area = BsplineArea()
    bs_area.show(data)
    # g = bs_area.grad(data)
    # data = data + 1e-5 * g
    # bs_area.show(data)
    # print(bs_area.eval(data), label[1])

    data = np.load('data/holes_cpt_train.npy')
    labels = np.load('data/labels_train.npy')
    bs_area = BsplineArea()
    rmae = []
    bar = tqdm(range(len(data)))
    for i in bar:
        data_i = data[i].reshape(-1)
        mass = bs_area.eval(data_i)
        rmae.append(np.abs(mass - labels[i, 1]) / labels[i, 1])
        bar.set_description('[%d/%d] rmae: %f' % (i, len(data), np.abs(mass - labels[i, 1]) / labels[i, 1]))

    rmae = np.array(rmae)
    ind = np.where(rmae<rmae.mean() + 3*rmae.std())[0]
    np.save('rmae_train_ind.npy', ind)
    print(rmae.mean(), len(rmae[rmae>0.01]), rmae.mean() + 3*rmae.std(), rmae[ind].argmax())
    plt.hist(rmae[rmae>0.01], bins=100)
    plt.show()
