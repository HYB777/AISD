import torch
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
from tqdm import tqdm
from xfoil import XFoil
from xfoil.model import Airfoil

INDEX = 0
CONFIG = 'config'


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
    def __init__(self, n, len_t=1000, d=3):
        super(BsplineArea, self).__init__()
        self.x_controls = np.hstack([0, (0.5 * (1 - np.cos(np.pi * np.arange(34 - 1).astype(np.double) / (34 - 2))))])

        self.name = 'BSArea'
        self.d = d
        self.dim = n
        self.dimOut = 1
        self.n = n
        np_float = np.float32
        self.N = np.zeros((len_t + 1, self.n), dtype=np_float)
        self.G = np.zeros((len_t, self.n - 1), dtype=np_float)
        self.NQ = np.zeros((self.n, self.n - 1), dtype=np_float)
        self.RR = np.zeros((self.n - 2, self.n - 2), dtype=np_float)
        self.RRFx = np.zeros((self.n - 2, self.n - 2), dtype=np_float)
        self.RRFy = np.zeros((self.n - 2, self.n - 2), dtype=np_float)
        self.t = np.zeros(self.n+d+1, dtype=np_float)
        for i in range(self.n+d+1):
            if i <= d:
                self.t[i] = 0
            elif d+1 <= i <= self.n - 1:
                self.t[i] = (i - d) / (self.n + 1 -d)
            else:
                self.t[i] = 1
        u = np.linspace(0, 1, len_t+1, dtype=np_float)
        self.u = (u[1:] + u[:-1]) / 2.
        self.du = 1 / len_t

        # self.init_NG()
        # self.init_N()

        try:
            self.NQ = np.load(CONFIG + '/NQ.npy')
            self.M = np.load(CONFIG + '/M.npy')
            self.RR = np.load(CONFIG + '/RR.npy')
            self.RRFx = np.load(CONFIG + '/RRFx.npy')
            self.RRFy = np.load(CONFIG + '/RRFy.npy')
        except FileNotFoundError:
            self.init_NG()

        try:
            self.N = np.load(CONFIG + '/Nbasis_%d.npy' % len(self.u))
        except FileNotFoundError:
            self.init_N()
        self.H = np.zeros((self.dim, self.dim), dtype=np_float)
        # self.H[self.dim//2:, :self.dim//2] = self.M.transpose()
        # self.H[:self.dim//2, self.dim//2:] = self.M

        # self.H_fair = np.zeros((self.dim, self.dim), dtype=np_float)
        # self.H_fair[:self.dim//2, :self.dim//2] = self.RRFx + self.RRFx.transpose()
        # self.H_fair[self.dim//2:, self.dim//2:] = self.RRFy + self.RRFy.transpose()

    def init_NG(self):

        for i in range(self.n):
            for j in range(self.n - 1):
                factor = ((self.t[i + self.d + 1] - self.t[i]) / (self.d + 1)) * ((self.t[j + self.d + 1] - self.t[j + 1]) / self.d)
                self.NQ[i, j] = factor * T(i, self.d + 1, self.t, j + 1, self.d, self.t) * \
                                math.factorial(self.d + 1) * math.factorial(self.d) / math.factorial(self.d + self.d)
                if i < self.n - 2 and j < self.n - 2:
                    factorRR = ((self.t[i + self.d + 1] - self.t[i + 2]) / (self.d - 1)) * (
                                (self.t[j + self.d + 1] - self.t[j + 2]) / (self.d - 1))
                    self.RR[i, j] = factorRR * T(i + 2, self.d - 1, self.t, j + 2, self.d - 1, self.t) * \
                                    math.factorial(self.d - 1) * math.factorial(self.d - 1) / math.factorial(2 * self.d - 3)
        print(self.NQ)
        A = np.zeros((self.n - 1, self.n))
        A[:, :-1] -= np.eye(self.n - 1)
        A[:, 1:] += np.eye(self.n - 1)
        Te = np.diag(self.d / (self.t[self.d+1:-1]-self.t[1:self.n]))@A

        AA = np.zeros((self.n - 2, self.n - 1))
        AA[:, :-1] -= np.eye(self.n - 2)
        AA[:, 1:] += np.eye(self.n - 2)
        TeAA = np.diag((self.d - 1) / (self.t[self.d+1:-2]-self.t[2:self.n]))@AA

        I = np.eye(self.dim//2)
        If = np.eye(self.dim//2)[:, ::-1]
        # per1 = np.vstack([-If, I])
        # per2 = np.vstack([If, I])
        self.M = self.NQ@Te
        Rtranx = TeAA@Te
        Rtrany = TeAA@Te
        self.RRFx = Rtranx.transpose()@self.RR@Rtranx
        self.RRFy = Rtrany.transpose()@self.RR@Rtrany
        np.save(CONFIG + '/M.npy', self.M)
        np.save(CONFIG + '/NQ.npy', self.NQ)

        np.save(CONFIG + '/RR.npy', self.RR)
        np.save(CONFIG + '/RRFy.npy', self.RRFy)
        np.save(CONFIG + '/RRFx.npy', self.RRFx)

    def init_N(self):
        u = np.linspace(0, 1, len(self.u) + 1)

        for i in tqdm(range(len(self.u) + 1)):
            for j in range(self.n):
                self.N[i, j] = _basis_(j, self.d, u[i], self.t)
        np.save(CONFIG + '/Nbasis_%d.npy' % len(self.u), self.N)

    def sampling(self, x):

        if len(x) == 65:
            x = x[:-1]
        x_up = np.hstack([0, x[:32], 0])
        x_down = np.hstack([0, x[32:], 0])

        x_up = np.vstack([self.x_controls, x_up])
        x_down = np.vstack([self.x_controls, x_down])

        x_up = x_up.reshape(2, -1).transpose()
        x_down = x_down.reshape(2, -1).transpose()

        # plt.plot(x_up[:, 0], x_up[:, 1], 'ro-')
        # plt.plot(x_down[:, 0], x_down[:, 1], 'ro-')
        #
        # c_up = self.N@x_up
        # c_down = self.N@x_down
        #
        # plt.plot(c_up[:, 0], c_up[:, 1], 'g-')
        # plt.plot(c_down[:, 0], c_down[:, 1], 'g-')
        # plt.axis('equal')

        k0 = np.load(CONFIG + '/knots0.npy')
        k1 = np.load(CONFIG + '/knots1.npy')
        N0 = np.zeros((len(k0), self.n))
        N1 = np.zeros((len(k1), self.n))
        for i in range(len(k0)):
            for j in range(self.n):
                N0[i, j] = _basis_(j, self.d, k0[i], self.t)
                N1[i, j] = _basis_(j, self.d, k1[i], self.t)
        c_up = N0@x_up
        c_down = N1@x_down
        hat_data_upper = np.hstack([c_up[:, 0].reshape(-1, 1), c_up[:, 1].reshape(-1, 1)])
        hat_data_lower = np.hstack([c_down[:, 0].reshape(-1, 1), c_down[:, 1].reshape(-1, 1)])
        hat_data = np.vstack([hat_data_upper[:-1], hat_data_lower])

        return hat_data, x_down, x_up

    def show(self, x, i=0, filename=None):
        x_copy = np.copy(x)
        alfa = x[-1]

        if len(x) == 65:
            x = x[:-1]
        x_up = np.hstack([0, x[:32], 0])
        x_down = np.hstack([0, x[32:], 0])

        x_up = np.vstack([self.x_controls, x_up])
        x_down = np.vstack([self.x_controls, x_down])

        x_up = x_up.reshape(2, -1).transpose()
        x_down = x_down.reshape(2, -1).transpose()

        plt.close()
        # plt.clf()
        plt.figure(i)

        plt.plot(x_up[:, 0], x_up[:, 1], 'ro-')
        plt.plot(x_down[:, 0], x_down[:, 1], 'ro-')

        c_up = self.N@x_up
        c_down = self.N@x_down

        plt.plot(c_up[:, 0], c_up[:, 1], 'g-')
        plt.plot(c_down[:, 0], c_down[:, 1], 'g-')
        plt.axis('equal')

        k0 = np.load(CONFIG + '/knots0.npy')
        k1 = np.load(CONFIG + '/knots1.npy')
        N0 = np.zeros((len(k0), self.n))
        N1 = np.zeros((len(k1), self.n))
        for i in range(len(k0)):
            for j in range(self.n):
                N0[i, j] = _basis_(j, self.d, k0[i], self.t)
                N1[i, j] = _basis_(j, self.d, k1[i], self.t)
        c_up = N0@x_up
        c_down = N1@x_down
        hat_data_upper = np.hstack([c_up[:, 0].reshape(-1, 1), c_up[:, 1].reshape(-1, 1)])
        hat_data_lower = np.hstack([c_down[:, 0].reshape(-1, 1), c_down[:, 1].reshape(-1, 1)])
        hat_data = np.vstack([hat_data_upper[:-1], hat_data_lower])
        # print(hat_data)
        xf = XFoil()
        xf.reset_bls()
        xf.airfoil = Airfoil(hat_data[:, 0], hat_data[:, 1])
        xf.repanel(cte_ratio=1)
        xf.Re = 6.5e6
        xf.M = 0.
        xf.max_iter = 100
        res_ = xf.a(alfa)
        print('cl: ', res_[0], 'cd: ', res_[1], 'alfa: ', alfa)

        if filename is None:

            plt.show()
        else:
            np.save('%s_c.npy' % filename, x_copy)
            np.save('%s.npy' % filename, hat_data)
            plt.savefig('%s.png' % filename)

    def eval(self, x):

        x_up = np.hstack([0, x[:32], 0])
        x_down = np.hstack([0, x[32:], 0])

        x_up = np.vstack([self.x_controls, x_up])
        x_down = np.vstack([self.x_controls, x_down])

        x_up = x_up.reshape(2, -1).transpose()
        x_down = x_down.reshape(2, -1).transpose()

        return x_up[:, 1]@self.M@x_up[:, 0] - x_down[:, 1]@self.M@x_down[:, 0]

    def grad(self, x):
        x_up = np.hstack([0, x[:32], 0])
        x_down = np.hstack([0, x[32:], 0])[::-1]

        x_up = np.vstack([self.x_controls, x_up])
        x_down = np.vstack([self.x_controls, x_down])

        x_up = x_up.reshape(2, -1).transpose()
        x_down = x_down.reshape(2, -1).transpose()

        gy_up = self.M@x_up[:, 0]
        gx_up = self.M.transpose()@x_up[:, 1]

        gy_down = self.M@x_down[:, 0]
        gx_down = self.M.transpose()@x_down[:, 1]

        return np.hstack([gy_up[1:-1], -gy_down[1:-1][::-1]])

    def hess(self, x):
        return self.H

    def preprocess_3d(self, x):
        sec = 64

        A_up = np.hstack([0, x[sec*0:sec*0+32], 0])
        A_down = np.hstack([0, x[sec*0+32:sec*1], 0])

        B_up = np.hstack([0, x[sec*1:sec*1+32], 0])
        B_down = np.hstack([0, x[sec*1+32:sec*2], 0])

        C_up = np.hstack([0, x[sec*2:sec*2+32], 0])
        C_down = np.hstack([0, x[sec*2+32:sec*3], 0])

        A_up = np.vstack([self.x_controls, A_up])
        A_down = np.vstack([self.x_controls, A_down])

        B_up = np.vstack([self.x_controls, B_up])
        B_down = np.vstack([self.x_controls, B_down])

        C_up = np.vstack([self.x_controls, C_up])
        C_down = np.vstack([self.x_controls, C_down])

        A_up = A_up.reshape(2, -1).transpose()
        A_down = A_down.reshape(2, -1).transpose()

        B_up = B_up.reshape(2, -1).transpose()
        B_down = B_down.reshape(2, -1).transpose()

        C_up = C_up.reshape(2, -1).transpose()
        C_down = C_down.reshape(2, -1).transpose()

        return A_up, A_down, B_up, B_down, C_up, C_down

    def eval_vol(self, x):
        A_up, A_down, B_up, B_down, C_up, C_down = self.preprocess_3d(x)

        Area_A = A_up[:, 1] @ self.M @ A_up[:, 0] - A_down[:, 1] @ self.M @ A_down[:, 0]
        Area_B = B_up[:, 1] @ self.M @ B_up[:, 0] - B_down[:, 1] @ self.M @ B_down[:, 0]
        Area_C = C_up[:, 1] @ self.M @ C_up[:, 0] - C_down[:, 1] @ self.M @ C_down[:, 0]

        V = 5.5 * Area_A + 6.75 * Area_B + 1.75 * Area_C

        return V

    def grad_vol(self, x):
        A_up, A_down, B_up, B_down, C_up, C_down = self.preprocess_3d(x)

        gA = np.hstack([(self.M @ A_up[:, 0])[1:-1], -(self.M @ A_down[:, 0])[1:-1]])
        gB = np.hstack([(self.M @ B_up[:, 0])[1:-1], -(self.M @ B_down[:, 0])[1:-1]])
        gC = np.hstack([(self.M @ C_up[:, 0])[1:-1], -(self.M @ C_down[:, 0])[1:-1]])

        return np.hstack([gA*5.5, 6.75*gB, 1.75*gC, 0])


if __name__ == '__main__':
    import torch.nn.functional as F
    data = np.load('extend_data_wgan/case_%d.npy' % INDEX)[1, :64]
    bs_area = BsplineArea(34)
    bs_area.show(data)
    print(bs_area.eval(data))
    # data_x = np.load('minA_conF/res_scale_16_vm1612102.8750_area228.7500.npy').astype(np.float32)
    # Tt = 100000
    # bs_area = BsplineArea(64, len_t=Tt)
    # c = bs_area.show(data_x, 200, filename='sdfsdqq')
    # sx, sy = c.shape
    #
    # lap = torch.tensor([1.0, -2.0, 1.0]).view(1, 1, 3).float()
    # c_torch = torch.from_numpy(c.transpose()).unsqueeze(0).float()
    # cppx = F.conv1d(c_torch[:, 0].unsqueeze(0), lap, stride=1, padding=0).detach().numpy()
    # cppy = F.conv1d(c_torch[:, 1].unsqueeze(0), lap, stride=1, padding=0).detach().numpy()
    # cpp = np.zeros((sx - 2, 2))
    # cpp[:, 0] = cppx[0, 0]
    # cpp[:, 1] = cppy[0, 0]

    # A = np.zeros((sx - 2, sx))
    # A[:, :sx - 2] += np.eye(sx - 2)
    # A[:, 1:sx - 1] += -2 * np.eye(sx - 2)
    # A[:, 2:] += np.eye(sx - 2)
    # dt = 1 / Tt
    # cpp = A @ c
    # cpp_ = (cpp / (dt**2)).astype(np.float32)
    # print(np.sum(cpp**2)/(dt**3))
    # print(bs_area.fair_eval(np.hstack([data_x, 0])))
    # print(tb-ta, tc-tb)

