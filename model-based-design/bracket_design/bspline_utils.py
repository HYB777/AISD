import numpy as np
import math
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy.io as scio
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

INDEX = 0
CONFIG = 'config'
LEN_T = 1000

CONFIG_DICT = {
    'lY': 0.072,
    'lZ': 0.070,
    'bDia': 0.008,
    'dCmp': 0.04,
    'rC': 0.004,
    'zC': 0.021,
    'sZ': 0.02,
    'thk': 0.003 / 5,
}


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


class BsplineArea:
    def __init__(self, n=34, d=3):

        '''
        :param n: the number of control points is n + 1
        :param d: degree
        '''

        super(BsplineArea, self).__init__()

        theta_cmp = np.linspace(1, 1.5, 100) * np.pi
        self.Cmp = np.vstack([np.cos(theta_cmp) * CONFIG_DICT['dCmp'] / 2. + CONFIG_DICT['lY'] / 2,
                              np.sin(theta_cmp) * CONFIG_DICT['dCmp'] / 2. + CONFIG_DICT['lZ']])
        theta_bdia = np.linspace(0, 2, 100) * np.pi
        self.bDia = np.vstack([np.cos(theta_bdia) * CONFIG_DICT['bDia'] / 2. + CONFIG_DICT['bDia'],
                               np.sin(theta_bdia) * CONFIG_DICT['bDia'] / 2. + CONFIG_DICT['lZ'] - CONFIG_DICT['bDia']])
        theta_cly6 = np.linspace(np.pi / 2, 1.5 * np.pi-np.arccos((CONFIG_DICT['zC'] - CONFIG_DICT['sZ']) / CONFIG_DICT['rC']), 100)
        self.cyl6 = np.vstack([np.cos(theta_cly6) * CONFIG_DICT['rC'] + CONFIG_DICT['lY'] / 2,
                               np.sin(theta_cly6) * CONFIG_DICT['rC'] + CONFIG_DICT['zC']])
        temp_dy = (CONFIG_DICT['rC']**2 - (CONFIG_DICT['zC'] - CONFIG_DICT['sZ'])**2)**0.5
        self.bbox1 = np.array([
            [CONFIG_DICT['lY'] / 2 - CONFIG_DICT['dCmp'] / 2, CONFIG_DICT['lZ']],
            [0, CONFIG_DICT['lZ']],
            [0, CONFIG_DICT['sZ']],
            [CONFIG_DICT['lY'] / 2 - temp_dy, CONFIG_DICT['sZ']]
        ])
        self.bbox2 = np.array([
            [CONFIG_DICT['lY'] / 2, CONFIG_DICT['zC'] + CONFIG_DICT['rC']],
            [CONFIG_DICT['lY'] / 2, CONFIG_DICT['lZ'] - CONFIG_DICT['dCmp'] / 2]
        ])

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

        # try:
        #     self.NN = np.load(CONFIG + '/NN.npy')
        # except FileNotFoundError:
        #     self.init_NN()

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

    # def init_NN(self):
    #     for i in range(self.n_control):
    #         for j in range(self.n_control):
    #             tlb = max(self.t[i], self.t[j + 1])
    #             tub = min(self.t[i + self.d + 1], self.t[j + self.d + 1])
    #             if tub <= tlb:
    #                 continue
    #             if self.t[self.d] <= tlb and tub <= self.t[self.n+1]:
    #                 factor = ((self.t[i + self.d + 1] - self.t[i]) / (self.d + 1)) * ((self.t[j + self.d + 1] - self.t[j]) / self.d)
    #                 self.NN[i, j] = factor * T(i, self.d + 1, self.t, j, self.d + 1, self.t) * \
    #                                 math.factorial(self.d + 1) * math.factorial(self.d + 1) / math.factorial(self.d + self.d + 1)
    #             else:
    #                 a = max(self.t[self.d], tlb)
    #                 b = min(self.t[self.n+1], tub)
    #                 if b > a:
    #                     txi = lambda xi: (b - a) / 2 * xi + (b + a) / 2
    #                     self.NN[i, j] = (np.array([_basis_(i, self.d, ti, self.t)*_basis_(j, self.d, ti, self.t)
    #                                                for ti in txi(self.gaussian_points)]) * self.gaussian_weight).sum() * (b - a) / 2
    #
    #     np.save(CONFIG + '/NN.npy', self.NN)

    def show(self, x, filename=None, curve_ref=None, message=None):

        P = x.reshape(2, -1).transpose()
        P = np.vstack([P, P[:self.d]])

        plt.plot(P[:, 0], P[:, 1], 'ro-', label='Bspline control points')
        # for i in range(len(P)):
        #     if i < len(P) - self.d:
        #         plt.text(P[i, 0], P[i, 1], 'P_%d' % i)
        #     else:
        #         plt.text(P[i, 0], P[i, 1], '     (P_%d)' % i)

        c = self.N@P

        plt.plot(c[:, 0], c[:, 1], 'g-', label='curve')
        plt.plot(self.Cmp[0], self.Cmp[1], 'k-')
        plt.plot(self.bDia[0], self.bDia[1], 'k-')
        plt.plot(self.cyl6[0], self.cyl6[1], 'k-')
        plt.plot(self.bbox1[:, 0], self.bbox1[:, 1], 'k-')
        plt.plot(self.bbox2[:, 0], self.bbox2[:, 1], 'k-')
        if curve_ref is not None:
            plt.plot(curve_ref[:, 0], curve_ref[:, 1], 'b.-', label='ref curve')
            mass_pt = curve_ref.mean(0)
            s = 0
            vec = curve_ref - mass_pt
            for i in range(len(curve_ref) - 1):
                s += abs(vec[i, 0]*vec[i + 1, 1] - vec[i, 1] * vec[i + 1, 0]) * 0.5
            print('refer area is ', s)
        else:
            mass_pt = c.mean(0)
            s = 0
            vec = c - mass_pt
            for i in range(len(c) - 1):
                s += abs(vec[i, 0]*vec[i + 1, 1] - vec[i, 1] * vec[i + 1, 0]) * 0.5
            print('refer area is ', s)

        plt.axis('equal')
        plt.legend()

        if filename is None:
            # print('...')
            plt.show()
        else:
            plt.savefig('%s.svg' % filename)
            plt.clf()
            np.save('%s_cpt.npy' % filename, x)
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

        Px = x[:32]
        Py = x[32:]

        # S = self.get_S(x)
        # M = self.M * S
        # MT = M @ self.T
        # WtMTMTtW = self.Warp.transpose() @ (MT - MT.transpose()) @ self.Warp

        # return 0.5 * Px @ WtMTMTtW @ Py
        return -0.5 * Px @ self.WtMTMTtW @ Py

    def grad(self, x):
        Px = x[:32]
        Py = x[32:]

        # S = self.get_S(x)
        # M = self.M * S
        # MT = M @ self.T
        # WtMTMTtW = self.Warp.transpose() @ (MT - MT.transpose()) @ self.Warp

        # gx = 0.5 * WtMTMTtW @ Py
        # gy = 0.5 * Px @ WtMTMTtW

        gx = -0.5 * self.WtMTMTtW @ Py
        gy = -0.5 * Px @ self.WtMTMTtW

        return np.hstack([gx, gy])

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
    bs_area = BsplineArea()
    print(np.linalg.eigvals(bs_area.WtMTMTtW))
    ind = 168350
    data = np.load('config/cpt_train.npy').transpose((0, 2, 1))[ind].reshape(-1)
    curve = np.load('config/params_train.npy')[ind]
    bs_area = BsplineArea()
    bs_area.show(data, curve_ref=None)
    g = bs_area.grad(data)
    data1 = data - g
    # plt.plot(data1[:32], data1[32:], 'ko-')
    # plt.show()
    print(bs_area.eval(data))
    print(bs_area.fair_eval(data))

    # input('......')
    # bs_area = BsplineArea()
    # area_train = np.load('config/area_train.npy')
    # ind_train = area_train > 0
    # area_train = area_train[ind_train]
    # data = np.load('config/cpt_train.npy').transpose((0, 2, 1)).reshape(-1, 64)
    # data = data[ind_train]
    # labels = np.load('config/labels_train.npy')
    # labels = labels[ind_train]
    # fair = np.array([bs_area.fair_eval(data[i]) for i in range(len(data))])
    # plt.hist(fair[::10], bins=100)
    #
    # print(fair.mean(), fair.mean() + fair.std(), fair.mean() + 2 * fair.std(), fair.mean() + 3 * fair.std())
    # print(fair.argmin())
    # plt.show()

    # input('.....')
    # from shapeOpt import BracketRequire
    # br = BracketRequire()
    # bs_area = BsplineArea()
    # PHASE = 'test'
    # data = np.load('config/cpt_%s.npy' % PHASE).transpose((0, 2, 1)).reshape(-1, 64)
    # areas = np.zeros(len(data))
    # for i in tqdm(range(len(data))):
    #     data_i = data[i]
    #     if br.eval(data_i) == 0:
    #         areas[i] = bs_area.eval(data_i)
    #     else:
    #         print(i)
    #         areas[i] = -1
    #
    # np.save('config/area_%s.npy' % PHASE, areas)
