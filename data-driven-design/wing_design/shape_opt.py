import os
import matplotlib.pyplot as plt
from resnet1d import *
from ae_nets import *
from bspline_utils import BsplineArea
from pyoptsparse import SLSQP, SNOPT, Optimization, PSQP, NLPQLP, IPOPT
import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as pynp
import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from wing_gen_script import generate_config_, generate_config_2
from xfoil import XFoil
from xfoil.model import Airfoil as XAirfoil
from convet2xml import convert2xml
from flightcondition import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--item', type=int, default=13392338, help='data item')

N_PARAMS = 14
RESULT_DIR = 'optimize_results/minCD_conCL_netALSSL'
VELOCITY = 236.
ALTITUDE = 12300.
LDRATIO = 86


def compute_airfoil(data, alfa, re, max_iter, c):
    xf = XFoil()
    fc = FlightCondition(h=12300 * unit('m'), TAS=236 * unit('m/s'), L=c * unit('m'))
    bunching_ = np.linspace(0.5, 10, 20)
    n_nodes_ = np.hstack([np.arange(160, 370, 10), np.arange(80, 160, 10)[::-1]])
    cte_ratio_ = np.hstack([np.linspace(1, 0.1, 37), np.linspace(1, 2, 41)[1:]])
    res, x, y, cp = None, None, None, None

    xf.reset_bls()
    xf.airfoil = XAirfoil(data[:, 0], data[:, 1])
    # xf.repanel(n_nodes=n_nodes, cte_ratio=cte_ratio, cv_par=bunching)
    xf.repanel(cv_par=2, cte_ratio=0., xt_ref=(0.8, 1), xb_ref=(0.8, 1), ctr_ratio=0.5)
    xf.Re = re
    # xf.Re = fc.Re
    xf.M = 0.7984694856192647  # 236 / 340
    xf.max_iter = max_iter
    res = xf.a(alfa)
    # res = xf.aseq(0, alfa, 100)
    x, y, cp = xf.get_cp_distribution()
    # res, x, y, cp = res[-1], x[-1], y[-1], cp[-1]
    # for n_nodes in n_nodes_:
    #     for bunching in bunching_:
    #         for cte_ratio in cte_ratio_:
    #             xf.reset_bls()
    #             xf.airfoil = XAirfoil(data[:, 0], data[:, 1])
    #             xf.repanel(n_nodes=n_nodes, cte_ratio=cte_ratio, cv_par=bunching)
    #             xf.Re = re
    #             xf.M = 0.7984694856192647  # 236 / 340
    #             xf.max_iter = max_iter
    #             res = xf.a(alfa)
    #             x, y, cp = xf.get_cp_distribution()
    #             if not np.any(np.isnan(cp)):
    #                 return res, x, y, cp
    return res, x, y, cp


def mirror_(pos):
    return [pos[0], -pos[1], pos[2]]


def wing_sec(airfoil, para_, use_mirror=False):
    if use_mirror:
        return asb.WingXSec(
            xyz_le=mirror_(para_[0]),
            chord=para_[1],
            twist=para_[2],
            airfoil=airfoil
        )
    else:
        return asb.WingXSec(
            xyz_le=para_[0],
            chord=para_[1],
            twist=para_[2],
            airfoil=airfoil
        )


class AENet:
    def __init__(self, ae, mu, sig):
        self.ae = ae
        self.mu = mu
        self.sig = sig

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def eval_(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x[:-N_PARAMS].reshape(5, 2, -1))
        recon = self.ae(xn)
        return (torch.abs(xn - recon).mean(dim=(1, 2))).detach().numpy().reshape(-1)

    def eval_i(self, x, i):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x[:-N_PARAMS].reshape(5, 1, 2, -1))
        recon = self.ae(xn[i])
        return (torch.abs(xn[i] - recon).mean()).detach().numpy().reshape(-1)

    def grad_i(self, x, i):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch[:-N_PARAMS].reshape(5, 1, 2, -1))
        recon = self.ae(xn[i])
        y = torch.abs(xn[i] - recon).mean()
        y.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy

    def eval(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x[:-N_PARAMS].reshape(5, 2, -1))
        recon = self.ae(xn)
        return (torch.abs(xn - recon).mean()).detach().numpy().reshape(-1)

    def grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch[:-N_PARAMS].reshape(5, 2, -1))
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

    def aspect_ratio_(self, x_torch):
        alfa, cr, ct, wl, span, theta, phi, psi, gamma, twist_r, twist_30, twist_70, twist_t, twist_w = x_torch[
                                                                                                        -N_PARAMS:]

        theta, phi, psi, gamma = theta * np.pi / 180, phi * np.pi / 180, psi * np.pi / 180, gamma * np.pi / 180

        tan_theta = torch.tan(theta)
        tan_phi = torch.tan(phi)

        yt = span
        zt = yt * tan_phi

        ys30 = 0.3 * yt
        cs30 = (cr - (tan_theta * ys30 + cr / 4)) * 4 / 3

        cw = 0.5 * ct
        yw = yt + wl
        zw = yw * tan_phi

        psi_ = torch.pi / 2 - psi - phi
        cos_psi_ = torch.cos(psi_)
        sin_psi_ = torch.sin(psi_)
        ywt = cos_psi_ * (yw - yt) - sin_psi_ * (zw - zt)
        yw = yt + ywt

        b = ((cr + cs30) / 2 * ys30 + (cs30 + ct) / 2 * (yt - ys30) + (ct + cw) / 2 * (yw - yt)) / yw
        return yw / b

    def aspect_ratio_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        lbr = self.aspect_ratio_(x_torch)
        return lbr.detach().numpy().reshape(-1)

    def aspect_ratio_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)
        lbr = self.aspect_ratio_(x_torch)
        lbr.backward()
        gxy = x_torch.grad.data.numpy().reshape(1, -1)
        return gxy

    def eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        cl, cd, vol = self.net(x_torch[:-N_PARAMS].reshape(1, 5 * 2, -1),
                               (x_torch[-N_PARAMS:] - self.mu_paras) / self.sig_paras)
        return cl.detach().numpy().reshape(-1), cd.detach().numpy().reshape(-1), vol.detach().numpy().reshape(-1)

    def cl_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        cl, cd, vol = self.net(x_torch[:-N_PARAMS].reshape(1, 5 * 2, -1),
                               (x_torch[-N_PARAMS:] - self.mu_paras) / self.sig_paras)
        return cl.detach().numpy().reshape(-1)

    def cd_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        cl, cd, vol = self.net(x_torch[:-N_PARAMS].reshape(1, 5 * 2, -1),
                               (x_torch[-N_PARAMS:] - self.mu_paras) / self.sig_paras)
        return cd.detach().numpy().reshape(-1)

    def vol_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        cl, cd, vol = self.net(x_torch[:-N_PARAMS].reshape(1, 5 * 2, -1),
                               (x_torch[-N_PARAMS:] - self.mu_paras) / self.sig_paras)
        return vol.detach().numpy().reshape(-1)

    def cl_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:-N_PARAMS].reshape(1, 5 * 2, -1),
                           (x_torch[-N_PARAMS:] - self.mu_paras) / self.sig_paras)
        y_torch[0].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def cd_grad(self, x):
        self.gradient_steps = self.gradient_steps + 1
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:-N_PARAMS].reshape(1, 5 * 2, -1),
                           (x_torch[-N_PARAMS:] - self.mu_paras) / self.sig_paras)
        y_torch[1].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def vol_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:-N_PARAMS].reshape(1, 5 * 2, -1),
                           (x_torch[-N_PARAMS:] - self.mu_paras) / self.sig_paras)
        y_torch[2].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy


class ShapeOptConPts:
    def __init__(self, cfd, ae, cl_scale, cd_scale, vol_scale):
        self.cfd = cfd
        self.ae = ae

        self.cl_scale = cl_scale
        self.cd_scale = cd_scale
        self.vol_scale = vol_scale

        self.bs = BsplineArea(34)

        I = np.eye(32)
        O = np.zeros((32, 32))
        self.A = np.zeros((32 * 5, 64 * 5 + N_PARAMS))
        A_ = np.zeros((32 * 5, 64 * 5))
        A_[:32, :64] = np.hstack([I, -I])
        A_[32:32 * 2, 64:64 * 2] = np.hstack([I, -I])
        A_[32 * 2:32 * 3, 64 * 2:64 * 3] = np.hstack([I, -I])
        A_[32 * 3:32 * 4, 64 * 3:64 * 4] = np.hstack([I, -I])
        A_[32 * 4:, 64 * 4:] = np.hstack([I, -I])
        self.A[:, :-N_PARAMS] = A_

        self.B = np.zeros((32 * 5, 64 * 5 + N_PARAMS))
        B_ = np.zeros((32 * 5, 64 * 5))
        B_[:32, :64] = np.hstack([I, O])
        B_[32:32 * 2, 64:64 * 2] = np.hstack([I, O])
        B_[32 * 2:32 * 3, 64 * 2:64 * 3] = np.hstack([I, O])
        B_[32 * 3:32 * 4, 64 * 3:64 * 4] = np.hstack([I, O])
        B_[32 * 4:, 64 * 4:] = np.hstack([I, O])
        self.B[:, :-N_PARAMS] = B_

        self.c1 = np.zeros((5, 64 * 5 + N_PARAMS))
        self.c2 = np.zeros((5, 64 * 5 + N_PARAMS))
        self.c3 = np.zeros((N_PARAMS, 64 * 5 + N_PARAMS))
        self.c1[0, 0] = 1
        self.c1[1, 64] = 1
        self.c1[2, 128] = 1
        self.c1[3, 192] = 1

        self.c2[0, 32] = 1
        self.c2[1, 32 + 64] = 1
        self.c2[2, 32 + 128] = 1
        self.c2[3, 32 + 192] = 1
        self.c2[4, 32 + 256] = 1

        self.c3[:, -N_PARAMS:] = np.eye(N_PARAMS)

        ALFA = [0, 3]
        ROOT_CHORD = [7.32, 7.88]
        TIP_CHORD = [1.25, 1.6]
        WINGLET_LEN = [2.4, 3.6]
        SPAN = [15, 18]
        CHORD_SWEEP = [20, 30]
        DIHEDRAL = [0, 7]
        CANT = [0, 45]
        WINGLET_SWEEP = [0, 50]
        ROOT_TWIST = [1, 4]
        SPAN30_TWIST = [0, 1]
        SPAN70_TWIST = [-1, 0]
        TIP_TWIST = [-4, -1]
        WINGLET_TWIST = [-5, 5]

        PARAMS_RANGE = [ALFA,
                        ROOT_CHORD, TIP_CHORD, WINGLET_LEN, SPAN,
                        CHORD_SWEEP, DIHEDRAL, CANT, WINGLET_SWEEP,
                        ROOT_TWIST, SPAN30_TWIST, SPAN70_TWIST, TIP_TWIST, WINGLET_TWIST]
        self.PARAMS_LB = [it[0] for it in PARAMS_RANGE]
        self.PARAMS_UB = [it[1] for it in PARAMS_RANGE]

        self.area_hist = []
        self.cl_hist = []
        self.cd_hist = []
        self.clcd_hist = []
        self.ae_hist = []
        self.Niter = 0
        self.alfa = None

    def draw_angle(self, theta, r):
        flag = 1 if theta > 0 else -1
        theta = -theta * np.pi / 180
        x0, y0 = r, 0
        x1, y1 = r * np.cos(theta), r * np.sin(theta)

        ox, oy = (x0 + x1) / 2, (y0 + y1) / 2
        r_ = 0.5 * ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        theta0 = flag * (np.pi - np.abs(theta)) / 2
        theta1 = theta0 - np.pi if flag == 1 else theta0 + np.pi

        phi = np.linspace(theta0, theta1, 100)
        x = ox + r_ * np.cos(phi)
        y = oy + r_ * np.sin(phi)
        return x, y

    def set_axes_equal(self, ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def draw_wings_mesh(self, a, b, c, d, e, paras):

        airfoil_a = asb.Airfoil(coordinates=a)
        airfoil_b = asb.Airfoil(coordinates=b)
        airfoil_c = asb.Airfoil(coordinates=c)
        airfoil_d = asb.Airfoil(coordinates=d)
        airfoil_e = asb.Airfoil(coordinates=e)

        alfa, para_a, para_b, para_c, para_d, para_e = generate_config_(paras)

        wing_name = 'opt'

        airplane = asb.Airplane(
            name=wing_name,
            xyz_ref=[0, 0, 0],  # CG location
            wings=[
                asb.Wing(
                    name="Main Wing",
                    symmetric=False,
                    xsecs=[
                        wing_sec(airfoil_a, para_a, False),
                        wing_sec(airfoil_b, para_b, False),
                        wing_sec(airfoil_c, para_c, False),
                        wing_sec(airfoil_d, para_d, False),
                        wing_sec(airfoil_e, para_e, False),
                    ]
                ),
            ],
        )

        points, faces = airplane.mesh_body(method="quad", thin_wings=False)

        ax = Axes3D(plt.figure())
        for f in faces:
            vtx = np.array([points[f[k]] for k in range(4)])
            tri = Poly3DCollection([vtx])
            tri.set_edgecolor(None)
            tri.set_facecolor('b')
            tri.set_alpha(0.3)
            ax.add_collection3d(tri)

        inda = np.abs(points[:, 1] - para_a[0][1]) < 0.1
        indb = np.abs(points[:, 1] - para_b[0][1]) < 0.1
        indc = np.abs(points[:, 1] - para_c[0][1]) < 0.1
        indd = np.abs(points[:, 1] - para_d[0][1]) < 0.1
        inde = np.abs(points[:, 1] - para_e[0][1]) < 0.1

        tria = Poly3DCollection([points[inda]])
        tria.set_edgecolor('r')
        tria.set_facecolor('r')
        ax.add_collection3d(tria)

        trib = Poly3DCollection([points[indb]])
        trib.set_edgecolor('r')
        trib.set_facecolor('r')
        ax.add_collection3d(trib)

        tric = Poly3DCollection([points[indc]])
        tric.set_edgecolor('r')
        tric.set_facecolor('r')
        ax.add_collection3d(tric)

        trid = Poly3DCollection([points[indd]])
        trid.set_edgecolor('r')
        trid.set_facecolor('r')
        ax.add_collection3d(trid)

        trie = Poly3DCollection([points[inde]])
        trie.set_edgecolor('r')
        trie.set_facecolor('r')
        ax.add_collection3d(trie)

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        self.set_axes_equal(ax)
        ax.view_init(elev=90, azim=0)
        plt.show()

    def draw_wings(self, x):
        a, a_down, a_up = self.bs.sampling(x[:64])
        b, b_down, b_up = self.bs.sampling(x[64:128])
        c, c_down, c_up = self.bs.sampling(x[128:192])
        d, d_down, d_up = self.bs.sampling(x[192:256])
        e, e_down, e_up = self.bs.sampling(x[256:-N_PARAMS])

        # twist_a = self.get_rot_mat(x[-N_PARAMS:][9])
        # twist_b = self.get_rot_mat(x[-N_PARAMS:][10])
        # twist_c = self.get_rot_mat(x[-N_PARAMS:][11])
        # twist_d = self.get_rot_mat(x[-N_PARAMS:][12])
        # twist_e = self.get_rot_mat(x[-N_PARAMS:][13])
        #
        # a_, a_down_, a_up_ = a @ twist_a.transpose(), a_down @ twist_a.transpose(), a_up @ twist_a.transpose()
        # b_, b_down_, b_up_ = b @ twist_b.transpose(), b_down @ twist_b.transpose(), b_up @ twist_b.transpose()
        # c_, c_down_, c_up_ = c @ twist_c.transpose(), c_down @ twist_c.transpose(), c_up @ twist_c.transpose()
        # d_, d_down_, d_up_ = d @ twist_d.transpose(), d_down @ twist_d.transpose(), d_up @ twist_d.transpose()
        # e_, e_down_, e_up_ = e @ twist_e.transpose(), e_down @ twist_e.transpose(), e_up @ twist_e.transpose()

        self.draw_wings_mesh(a, b, c, d, e, x[-N_PARAMS:])

    def draw_airfoils(self, x):
        filename = 'figures_for_paper'
        a, a_down, a_up = self.bs.sampling(x[:64])
        b, b_down, b_up = self.bs.sampling(x[64:128])
        c, c_down, c_up = self.bs.sampling(x[128:192])
        d, d_down, d_up = self.bs.sampling(x[192:256])
        e, e_down, e_up = self.bs.sampling(x[256:-N_PARAMS])

        twist_a = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 5])
        twist_b = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 4] * 4)
        twist_c = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 3] * 4)
        twist_d = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 2])
        twist_e = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 1])

        ang_a = self.draw_angle(x[-N_PARAMS:][N_PARAMS - 5], 0.5)
        ang_b = self.draw_angle(x[-N_PARAMS:][N_PARAMS - 4] * 4, 0.5)
        ang_c = self.draw_angle(x[-N_PARAMS:][N_PARAMS - 3] * 4, 0.5)
        ang_d = self.draw_angle(x[-N_PARAMS:][N_PARAMS - 2], 0.5)
        ang_e = self.draw_angle(x[-N_PARAMS:][N_PARAMS - 1], 0.5)

        a_, a_down_, a_up_ = a @ twist_a.transpose(), a_down @ twist_a.transpose(), a_up @ twist_a.transpose()
        b_, b_down_, b_up_ = b @ twist_b.transpose(), b_down @ twist_b.transpose(), b_up @ twist_b.transpose()
        c_, c_down_, c_up_ = c @ twist_c.transpose(), c_down @ twist_c.transpose(), c_up @ twist_c.transpose()
        d_, d_down_, d_up_ = d @ twist_d.transpose(), d_down @ twist_d.transpose(), d_up @ twist_d.transpose()
        e_, e_down_, e_up_ = e @ twist_e.transpose(), e_down @ twist_e.transpose(), e_up @ twist_e.transpose()
        COLOR = 'red'
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(5, 1, sharex='col', sharey='row', figsize=(7, 12))

        ha = a_[:, 1].max() - a_[:, 1].min()
        hb = b_[:, 1].max() - b_[:, 1].min()
        hc = c_[:, 1].max() - c_[:, 1].min()
        hd = d_[:, 1].max() - d_[:, 1].min()
        he = e_[:, 1].max() - e_[:, 1].min()

        ax[0].fill(a_[:, 0], a_[:, 1], facecolor=COLOR, edgecolor=COLOR, linewidth=1)
        ax[0].plot([0, a_[0, 0]], [0, a_[0, 1]], 'go--', linewidth=3)
        ax[0].plot(ang_a[0], ang_a[1], 'g-', linewidth=3)
        ax[0].arrow(0, 0, 1.2, 0, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[0].arrow(0, -0.6 * ha, 0, 1.2 * ha, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[0].set_title('root')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_axis_off()
        ax[0].axis('equal')

        ax[1].fill(b_[:, 0], b_[:, 1], facecolor=COLOR, edgecolor=COLOR, linewidth=1)
        ax[1].plot([0, b_[0, 0]], [0, b_[0, 1]], 'go--', linewidth=3)
        ax[1].plot(ang_b[0], ang_b[1], 'g-', linewidth=3)
        ax[1].arrow(0, 0, 1.2, 0, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[1].arrow(0, -0.6 * hb, 0, 1.2 * hb, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[1].set_title('span30%')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_axis_off()
        ax[1].axis('equal')

        ax[2].fill(c_[:, 0], c_[:, 1], facecolor=COLOR, edgecolor=COLOR, linewidth=1)
        ax[2].plot([0, c_[0, 0]], [0, c_[0, 1]], 'go--', linewidth=3)
        ax[2].plot(ang_c[0], ang_c[1], 'g-', linewidth=3)
        ax[2].arrow(0, 0, 1.2, 0, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[2].arrow(0, -0.6 * hc, 0, 1.2 * hc, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[2].set_title('span70%')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_axis_off()
        ax[2].axis('equal')

        ax[3].fill(d_[:, 0], d_[:, 1], facecolor=COLOR, edgecolor=COLOR, linewidth=1)
        ax[3].plot([0, d_[0, 0]], [0, d_[0, 1]], 'go--', linewidth=3)
        ax[3].plot(ang_d[0], ang_d[1], 'g-', linewidth=3)
        ax[3].arrow(0, 0, 1.2, 0, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[3].arrow(0, -0.6 * hd, 0, 1.2 * hd, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[3].set_title('tip')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].set_axis_off()
        ax[3].axis('equal')

        ax[4].fill(e_[:, 0], e_[:, 1], facecolor=COLOR, edgecolor=COLOR, linewidth=1)
        ax[4].plot([0, e_[0, 0]], [0, e_[0, 1]], 'go--', linewidth=3)
        ax[4].plot(ang_e[0], ang_e[1], 'g-', linewidth=3)
        ax[4].arrow(0, 0, 1.2, 0, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[4].arrow(0, -0.6 * he, 0, 1.2 * he, width=0.005, head_width=0.02, ec='black', fc='black')
        ax[4].set_title('winglet')
        ax[4].set_xticks([])
        ax[4].set_yticks([])
        ax[4].set_axis_off()
        ax[4].axis('equal')

        # plt.savefig(filename + '/airfoils_view.svg')
        plt.show()

    def print_paras(self, x0, xs):

        paras0 = x0[-N_PARAMS:]
        paras1 = xs[-N_PARAMS:]
        print(
            '\nalfa: ', paras0[0], ' --> ', paras1[0],
            '\nroot chord: ', paras0[1], ' --> ', paras1[1],
            '\ntip chord: ', paras0[2], ' --> ', paras1[2],
            '\nwinglet length: ', paras0[3], ' --> ', paras1[3],
            '\nspan: ', paras0[4], ' --> ', paras1[4],
            '\n1/4 chord sweep angle: ', paras0[5], ' --> ', paras1[5],
            '\ndihedral: ', paras0[6], ' --> ', paras1[6],
            '\ncant angle: ', paras0[7], ' --> ', paras1[7],
            '\nwinglet sweep angle: ', paras0[8], ' --> ', paras1[8],
            '\nroot twist: ', paras0[9], ' --> ', paras1[9],
            '\nspan30 twist: ', paras0[10], ' --> ', paras1[10],
            '\nspan70 twist: ', paras0[11], ' --> ', paras1[11],
            '\ntip twist: ', paras0[12], ' --> ', paras1[12],
            '\nwinglet twist: ', paras0[13], ' --> ', paras1[13],
        )

    def cal_vol(self, airplane):
        points, faces = airplane.mesh_body(method="quad", thin_wings=False)
        v = 0
        for f in faces:
            p0, p1, p2, p3 = points[f[0]], points[f[1]], points[f[2]], points[f[3]]

            p01 = p1 - p0
            p02 = p2 - p0
            p03 = p3 - p0
            n012 = np.cross(p01, p02)
            n023 = np.cross(p02, p03)
            cent012 = (p0 + p1 + p2) / 3.
            cent023 = (p0 + p2 + p3) / 3.

            v += 0.5 * (np.dot(cent012, n012) + np.dot(cent023, n023))
        return v / 6.0

    def get_rot_mat(self, twist_angle):
        twist_angle = twist_angle * np.pi / 180
        cos_psi_ = np.cos(twist_angle)
        sin_psi_ = np.sin(twist_angle)
        rot = np.array([[cos_psi_, sin_psi_],
                        [-sin_psi_, cos_psi_]])
        return rot

    def solve(self, x, filename=None):
        a, a_down, a_up = self.bs.sampling(x[:64])
        b, b_down, b_up = self.bs.sampling(x[64:128])
        c, c_down, c_up = self.bs.sampling(x[128:192])
        d, d_down, d_up = self.bs.sampling(x[192:256])
        e, e_down, e_up = self.bs.sampling(x[256:-N_PARAMS])

        twist_a = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 5])
        twist_b = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 4])
        twist_c = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 3])
        twist_d = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 2])
        twist_e = self.get_rot_mat(x[-N_PARAMS:][N_PARAMS - 1])

        a_, a_down_, a_up_ = a @ twist_a.transpose(), a_down @ twist_a.transpose(), a_up @ twist_a.transpose()
        b_, b_down_, b_up_ = b @ twist_b.transpose(), b_down @ twist_b.transpose(), b_up @ twist_b.transpose()
        c_, c_down_, c_up_ = c @ twist_c.transpose(), c_down @ twist_c.transpose(), c_up @ twist_c.transpose()
        d_, d_down_, d_up_ = d @ twist_d.transpose(), d_down @ twist_d.transpose(), d_up @ twist_d.transpose()
        e_, e_down_, e_up_ = e @ twist_e.transpose(), e_down @ twist_e.transpose(), e_up @ twist_e.transpose()

        plt.close()
        plt.clf()
        fig, ax = plt.subplots(5, 1, sharex='col', sharey='row', figsize=(7, 12))
        ax[0].plot(a_down_[:, 0], a_down_[:, 1], 'ro-')
        ax[0].plot(a_up_[:, 0], a_up_[:, 1], 'ro-')
        ax[0].plot(a_[:, 0], a_[:, 1], 'g')
        ax[0].set_title('root')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_axis_off()
        ax[0].axis('equal')

        ax[1].plot(b_down_[:, 0], b_down_[:, 1], 'ro-')
        ax[1].plot(b_up_[:, 0], b_up_[:, 1], 'ro-')
        ax[1].plot(b_[:, 0], b_[:, 1], 'g')
        ax[1].set_title('span30%')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_axis_off()
        ax[1].axis('equal')

        ax[2].plot(c_down_[:, 0], c_down_[:, 1], 'ro-')
        ax[2].plot(c_up_[:, 0], c_up_[:, 1], 'ro-')
        ax[2].plot(c_[:, 0], c_[:, 1], 'g')
        ax[2].set_title('span70%')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_axis_off()
        ax[2].axis('equal')

        ax[3].plot(d_down_[:, 0], d_down_[:, 1], 'ro-')
        ax[3].plot(d_up_[:, 0], d_up_[:, 1], 'ro-')
        ax[3].plot(d_[:, 0], d_[:, 1], 'g')
        ax[3].set_title('tip')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].set_axis_off()
        ax[3].axis('equal')

        ax[4].plot(e_down_[:, 0], e_down_[:, 1], 'ro-')
        ax[4].plot(e_up_[:, 0], e_up_[:, 1], 'ro-')
        ax[4].plot(e_[:, 0], e_[:, 1], 'g')
        ax[4].set_title('winglet')
        ax[4].set_xticks([])
        ax[4].set_yticks([])
        ax[4].set_axis_off()
        ax[4].axis('equal')

        plt.savefig(RESULT_DIR + '/' + filename + '_airfoils.svg')
        np.savetxt(RESULT_DIR + '/' + filename + '_root.dat', a)
        np.savetxt(RESULT_DIR + '/' + filename + '_span30.dat', b)
        np.savetxt(RESULT_DIR + '/' + filename + '_span70.dat', c)
        np.savetxt(RESULT_DIR + '/' + filename + '_tip.dat', d)
        np.savetxt(RESULT_DIR + '/' + filename + '_winglet.dat', e)
        np.savetxt(RESULT_DIR + '/' + filename + '_root_.dat', a_)
        np.savetxt(RESULT_DIR + '/' + filename + '_span30_.dat', b_)
        np.savetxt(RESULT_DIR + '/' + filename + '_span70_.dat', c_)
        np.savetxt(RESULT_DIR + '/' + filename + '_tip_.dat', d_)
        np.savetxt(RESULT_DIR + '/' + filename + '_winglet_.dat', e_)
        self.solve_(a, b, c, d, e, x[-N_PARAMS:], filename)
        np.save(RESULT_DIR + '/' + filename + '.npy', x)

    def solve_(self, a, b, c, d, e, paras, filename=None):
        airfoil_a = asb.Airfoil(coordinates=a)
        airfoil_b = asb.Airfoil(coordinates=b)
        airfoil_c = asb.Airfoil(coordinates=c)
        airfoil_d = asb.Airfoil(coordinates=d)
        airfoil_e = asb.Airfoil(coordinates=e)

        alfa, para_a, para_b, para_c, para_d, para_e = generate_config_(paras)

        wing_name = 'opt'

        airplane = asb.Airplane(
            name=wing_name,
            xyz_ref=[0, 0, 0],  # CG location
            wings=[
                asb.Wing(
                    name="Main Wing",
                    symmetric=False,
                    xsecs=[
                        wing_sec(airfoil_e, para_e, True),
                        wing_sec(airfoil_d, para_d, True),
                        wing_sec(airfoil_c, para_c, True),
                        wing_sec(airfoil_b, para_b, True),
                        wing_sec(airfoil_a, para_a, False),
                        wing_sec(airfoil_b, para_b, False),
                        wing_sec(airfoil_c, para_c, False),
                        wing_sec(airfoil_d, para_d, False),
                        wing_sec(airfoil_e, para_e, False),
                    ]
                ),
            ],
        )

        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                atmosphere=asb.Atmosphere(ALTITUDE),
                velocity=VELOCITY,  # m/s
                alpha=alfa,  # degree
            )
        )

        aero = vlm.run()
        print('CL: ', aero['CL'], 'CD: ', aero['CD'], 'VOL: ', self.cal_vol(airplane),
              'aspect: ', airplane.wings[0].aspect_ratio())

        if filename is not None:
            h = vlm.draw(show=False, colorbar_label='Delta Cp', c=vlm.vortex_strengths)
            h.save_graphic(RESULT_DIR + '/' + filename + '.svg')
        else:
            vlm.draw(colorbar_label='Delta Cp', c=vlm.vortex_strengths)

    def print_init(self, xk):
        area_k = self.cfd.vol_eval(xk) * self.vol_scale
        cl_k = self.cfd.cl_eval(xk) * self.cl_scale
        cd_k = self.cfd.cd_eval(xk) * self.cd_scale
        ae_k = self.ae.eval(xk)
        print('init area: ', area_k,
              'init cl: ', cl_k,
              'init cd: ', cd_k,
              'init cl/cd: ', cl_k / cd_k,
              'init ae: ', ae_k)
        print('+++++++++++++++++++++++++++++++++++++++++')

    def minCD_conCL(self, xdict):
        funcs = {}
        funcs['obj'] = self.cfd.cd_eval(xdict['xvars'])

        funcs['area'] = [
            self.cfd.vol_eval(xdict['xvars']).item(),
        ]
        funcs['aspect'] = [
            self.cfd.aspect_ratio_eval(xdict['xvars']).item(),
        ]

        funcs['con1'] = [
            self.A @ xdict['xvars'],
        ]
        funcs['con2'] = [
            self.cfd.cl_eval(xdict['xvars']).item(),
            self.ae.eval_i(xdict['xvars'], 0).item(),
            self.ae.eval_i(xdict['xvars'], 1).item(),
            self.ae.eval_i(xdict['xvars'], 2).item(),
            self.ae.eval_i(xdict['xvars'], 3).item(),
            self.ae.eval_i(xdict['xvars'], 4).item(),
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

    def minCD_conCL_sens(self, xdict, fdict):
        sens = {
            'obj': {'xvars': self.cfd.cd_grad(xdict['xvars'])},

            'area': {'xvars': self.cfd.vol_grad(xdict['xvars'])},
            'aspect': {'xvars': self.cfd.aspect_ratio_grad(xdict['xvars'])},

            'con2': {'xvars': np.vstack(
                [
                    self.cfd.cl_grad(xdict['xvars']),
                    self.ae.grad_i(xdict['xvars'], 0),
                    self.ae.grad_i(xdict['xvars'], 1),
                    self.ae.grad_i(xdict['xvars'], 2),
                    self.ae.grad_i(xdict['xvars'], 3),
                    self.ae.grad_i(xdict['xvars'], 4),
                ]
            )},
        }
        return sens

    def solve_minCD_conCL(self, x0):

        self.print_init(x0)
        ae_eps = 8e-4  # 8e-4, ae_mu - 3 * ae_sig

        A0 = self.cfd.vol_eval(x0)
        eA0 = 0.7 * A0
        EA0 = 1.2 * A0
        thick = self.A @ x0
        cls0 = self.cfd.cl_eval(x0).item() * self.cl_scale
        aspect0 = self.cfd.aspect_ratio_eval(x0)

        optProb = Optimization('shape opt', self.minCD_conCL)
        optProb.addVarGroup('xvars', 5 * 64 + N_PARAMS, value=x0)
        optProb.addConGroup('con1', 32 * 5, lower=thick * 0.0, linear=True, jac={'xvars': self.A})
        optProb.addConGroup('con2', 6, lower=[0.5 / self.cl_scale] + [None] * 5, upper=[None] + [ae_eps] * 5)
        optProb.addConGroup('area', 1, lower=eA0, upper=EA0)
        optProb.addConGroup('con3', 32 * 5, lower=np.zeros(32 * 5), linear=True, jac={'xvars': self.B})
        optProb.addConGroup('con4', 5, upper=0, linear=True, jac={'xvars': self.c2})
        optProb.addConGroup('con5', N_PARAMS, lower=self.PARAMS_LB, upper=self.PARAMS_UB, linear=True,
                            jac={'xvars': self.c3})
        # optProb.addConGroup('aspect', 1, lower=None, upper=aspect0.item())
        optProb.addObj('obj')
        print(optProb)
        optOption = {'IPRINT': -1, 'MIT': 1000}
        opt = PSQP(options=optOption)
        # optOption = {'max_iter': 1000}
        # opt = IPOPT(options=optOption)

        sol = opt(optProb, sens=self.minCD_conCL_sens)
        # sol = opt(optProb, sens='FD')
        print(sol)
        xs = np.array([v.value for v in sol.variables['xvars']])
        input('press anything to continue...')

        print('=====================================================================')
        cls0 = self.cfd.cl_eval(x0) * self.cl_scale
        cds0 = self.cfd.cd_eval(x0) * self.cd_scale
        cls1 = self.cfd.cl_eval(xs) * self.cl_scale
        cds1 = self.cfd.cd_eval(xs) * self.cd_scale
        print('\ninit area: ', self.cfd.vol_eval(x0) * self.vol_scale, ' --> opt area: ',
              self.cfd.vol_eval(xs) * self.vol_scale,
              '\ninit cl: ', cls0, ' --> opt cl: ', cls1,
              '\ninit cd: ', cds0, ' --> opt cd: ', cds1,
              '\ninit cl/cd: ', cls0 / cds0, ' --> opt cl/cd: ', cls1 / cds1,
              '\ninit ae: ', self.ae.eval_(x0), ' --> opt ae: ', self.ae.eval_(xs),
              '\ninit alfa: ', x0[-N_PARAMS], ' --> opt alfa: ', xs[-N_PARAMS],
              '\ninit aspect: ', self.cfd.aspect_ratio_eval(x0), ' --> opt aspect: ', self.cfd.aspect_ratio_eval(xs),
              )
        self.print_paras(x0, xs)
        # input('gradient steps: %d' % self.cfd.gradient_steps)
        # input('wait ...')
        self.solve(x0, 'init')
        self.solve(xs, 'opt')
        return xs

    def convert2xflr(self, prefix):
        x = np.load(RESULT_DIR + '/%s.npy' % prefix)
        path_prefix = RESULT_DIR.split('/')[-1] + '/' + prefix
        alfa, para_a, para_b, para_c, para_d, para_e = generate_config_2(x[-14:])
        xml_str = convert2xml(path_prefix,
                              para_a,
                              para_b,
                              para_c,
                              para_d,
                              para_e)
        f = open(RESULT_DIR + '/plane_%s.xml' % prefix, 'w', encoding='UTF-8')
        f.write(xml_str)
        print('angel of attack: ', alfa)
        print(para_a)
        print(para_b)
        print(para_c)
        print(para_d)
        print(para_e)
        print('----------------------------')
        return alfa, [para_a[-1], para_b[-1], para_c[-1], para_d[-1], para_e[-1]]

    def compare_airfoils(self, re_init, re_opt, alfa_init, alfa_opt, alfa_init_=0., alfa_opt_=0., c_init=None,
                         c_opt=None):
        _airfoils = ['_root.dat', '_span30.dat', '_span70.dat', '_tip.dat', '_winglet.dat']
        fig, (ax0, ax1) = plt.subplots(2, 5, gridspec_kw={'height_ratios': [2, 1]}, figsize=(50, 5))
        for i in range(5):
            _airfoil = _airfoils[i]
            init_re_i = re_init[i]
            opt_re_i = re_opt[i]
            alfa_init_i = alfa_init[i] + alfa_init_
            alfa_opt_i = alfa_opt[i] + alfa_opt_
            init_ = np.loadtxt(RESULT_DIR + '/init' + _airfoil)
            opt_ = np.loadtxt(RESULT_DIR + '/opt' + _airfoil)

            res_0, x0, y0, cp0 = compute_airfoil(data=init_, alfa=alfa_init_i, re=init_re_i, max_iter=100, c=c_init[i])
            res_1, x1, y1, cp1 = compute_airfoil(data=opt_, alfa=alfa_opt_i, re=opt_re_i, max_iter=100, c=c_opt[i])

            h0 = len(x0) // 2
            h1 = len(x1) // 2

            ax0[i].plot(x0[:h0], cp0[:h0], 'r-', linewidth=3, label='initial')
            ax0[i].plot(x0[h0:], cp0[h0:], 'r-', linewidth=3)
            ax0[i].plot(x1[:h1], cp1[:h1], 'g-', linewidth=3, label='optimized')
            ax0[i].plot(x1[h1:], cp1[h1:], 'g-', linewidth=3)
            ax0[i].invert_yaxis()
            # ax0.set_ylabel('$C_p$')

            ax0[i].set_xticks([])
            ax0[i].spines['top'].set_visible(False)
            ax0[i].spines['bottom'].set_visible(False)
            ax0[i].spines['right'].set_visible(False)
            ax1[i].plot(x0[:h0], y0[:h0], 'r-', linewidth=3, label='initial')
            ax1[i].plot(x0[h0:], y0[h0:], 'r-', linewidth=3)
            ax1[i].plot(x1[:h1], y1[:h1], 'g-', linewidth=3, label='optimized')
            ax1[i].plot(x1[h1:], y1[h1:], 'g-', linewidth=3)
            ax1[i].spines['top'].set_visible(False)
            ax1[i].spines['right'].set_visible(False)
        plt.legend(fontsize=15)
        plt.show()
        # plt.savefig(RESULT_DIR + '/airfoils_cp_%s.svg' % RESULT_DIR.split('_')[-1])
        plt.savefig('./airfoils_cp_%s.svg' % RESULT_DIR.split('_')[-1])

    def compute_re(self):
        x_init = np.load(RESULT_DIR + '/init.npy')
        _, para_a_init, para_b_init, para_c_init, para_d_init, para_e_init = generate_config_2(x_init[-14:])
        pos_init = [para_a_init[0], para_b_init[0], para_c_init[0], para_d_init[0], para_e_init[0]]
        c_init = [para_a_init[1], para_b_init[1], para_c_init[1], para_d_init[1], para_e_init[1]]

        x_opt = np.load(RESULT_DIR + '/opt.npy')
        _, para_a_opt, para_b_opt, para_c_opt, para_d_opt, para_e_opt = generate_config_2(x_opt[-14:])
        pos_opt = [para_a_opt[0], para_b_opt[0], para_c_opt[0], para_d_opt[0], para_e_opt[0]]
        c_opt = [para_a_opt[1], para_b_opt[1], para_c_opt[1], para_d_opt[1], para_e_opt[1]]

        re = np.loadtxt(RESULT_DIR + '/re.txt')
        re_init = re[:, :2]
        re_opt = re[:, 2:]

        re_init_res = []
        re_opt_res = []

        for i in range(5):
            if i == 0:
                # re_init_res.append(re_init[0, 1])
                # re_opt_res.append(re_opt[0, 1])
                re_init_res.append(
                    -(re_init[0, 1] - re_init[1, 1]) / (re_init[0, 0] - re_init[1, 0]) * re_init[1, 0] + re_init[1, 1])
                re_opt_res.append(
                    -(re_opt[0, 1] - re_opt[1, 1]) / (re_opt[0, 0] - re_opt[1, 0]) * re_opt[1, 0] + re_opt[1, 1])
            elif i == 4:
                # re_init_res.append(re_init[-1, 1])
                # re_opt_res.append(re_opt[-1, 1])
                re_init_res.append((re_init[-1, 1] - re_init[-2, 1]) / (re_init[-1, 0] - re_init[-2, 0]) * (
                            pos_init[i] - re_init[-2, 0]) + re_init[-2, 1])
                re_opt_res.append(
                    (re_opt[-1, 1] - re_opt[-2, 1]) / (re_opt[-1, 0] - re_opt[-2, 0]) * (pos_opt[i] - re_opt[-2, 0]) +
                    re_opt[-2, 1])
            else:
                pos_init_i = pos_init[i]
                pos_opt_i = pos_opt[i]
                ind_init = pynp.where((re_init[:, 0] <= pos_init_i))[0]
                ind_opt = pynp.where((re_opt[:, 0] <= pos_opt_i))[0]
                x_init_0, y_init_0 = re_init[ind_init[-1]]
                x_init_1, y_init_1 = re_init[ind_init[-1] + 1]
                x_opt_0, y_opt_0 = re_opt[ind_opt[-1]]
                x_opt_1, y_opt_1 = re_opt[ind_opt[-1] + 1]
                re_init_res.append(
                    (((x_init_1 - pos_init_i) * y_init_0 + (pos_init_i - x_init_0) * y_init_1) / (x_init_1 - x_init_0)))
                re_opt_res.append(
                    (((x_opt_1 - pos_opt_i) * y_opt_0 + (pos_opt_i - x_opt_0) * y_opt_1) / (x_opt_1 - x_opt_0)))
        return re_init_res, re_opt_res, c_init, c_opt

    def view_cl_icd_distribution(self):
        cl = np.loadtxt(RESULT_DIR + '/CL.txt')
        icd = np.loadtxt(RESULT_DIR + '/ICD.txt')

        x_init = np.load(RESULT_DIR + '/init.npy')
        _, para_a_init, para_b_init, para_c_init, para_d_init, para_e_init = generate_config_2(x_init[-14:])
        pos_init = [para_a_init[0], para_b_init[0], para_c_init[0], para_d_init[0], para_e_init[0]]

        x_opt = np.load(RESULT_DIR + '/opt.npy')
        _, para_a_opt, para_b_opt, para_c_opt, para_d_opt, para_e_opt = generate_config_2(x_opt[-14:])
        pos_opt = [para_a_opt[0], para_b_opt[0], para_c_opt[0], para_d_opt[0], para_e_opt[0]]

        cl_init, cl_opt = cl[np.abs(cl[:, 0]) < pos_init[3], :2], cl[np.abs(cl[:, 2]) < pos_opt[3], 2:]
        icd_init, icd_opt = icd[np.abs(icd[:, 0]) < pos_init[3], :2], icd[np.abs(icd[:, 0]) < pos_opt[3], 2:]
        fig, (ax0, ax1) = plt.subplots(2, 1)
        ax0.plot(cl_init[:, 0], cl_init[:, 1], 'r-')
        ax0.plot(cl_opt[:, 0], cl_opt[:, 1], 'g-')
        ax1.plot(icd_init[:, 0], icd_init[:, 1], 'r-')
        ax1.plot(icd_opt[:, 0], icd_opt[:, 1], 'g-')
        plt.savefig('./airfoils_cld_%s.svg' % RESULT_DIR.split('_')[-1])


if __name__ == '__main__':
    args = parser.parse_args()
    ITEM_DICT = {0: 'data_train_wings_rmmtw_new_0_100000',
                 1: 'data_train_wings_rmmtw_new_100000_200000',
                 2: 'data_train_wings_rmmtw_new_200000_300000',
                 3: 'data_train_wings_rmmtw_new_300000_400000',
                 4: 'data_train_wings_rmmtw_new_400000_500000'}

    ITEM = args.item
    RESULT_DIR_bk = RESULT_DIR
    RESULT_DIR = RESULT_DIR + '_%d' % ITEM
    try:
        os.mkdir(RESULT_DIR)
    except OSError:
        pass

    data = np.load('config/%s.npy' % ITEM_DICT[ITEM // 6400000], allow_pickle=True).item()
    ITEM = ITEM % 6400000
    wings_ = data['wings'][ITEM // 64].reshape(-1)
    paras_ = data['res'][ITEM][:N_PARAMS]
    x0 = np.hstack([wings_, paras_])

    net = ResNet1d18_RMMTW(use_bn=True, n_params=N_PARAMS)

    def _init_fn(worker_id):
        np.random.seed(1 + worker_id)


    check = torch.load('mean-teacher-al/results-al-semi/net_best_teacher.tar', map_location=torch.device('cpu'))
    mu_paras = check['option']['mu_paras']
    sig_paras = check['option']['sig_paras']
    net.load_state_dict(check['teacher'])
    net.cpu()
    net.eval()
    cfd = CFDNet(net=net, mu_paras=mu_paras, sig_paras=sig_paras)

    df = 4
    ae_net = AEcoder(ndf=df, ngf=df)
    check_ae = torch.load('../airfoil_design/shape-anomaly-detection/resultsAE_ID=16_wgan/ae_best_zeros.tar')
    ae_net.load_state_dict(check_ae['net_state_dict'])
    ae_net.cpu()
    ae_net.eval()
    ae = AENet(ae=ae_net, mu=check_ae['option']['mu'], sig=check_ae['option']['sig'])

    scales = check['scales']  # torch.ones_like(check['scales'])
    scale_cl = scales[0].item()
    scale_cd = scales[1].item()
    scale_vol = scales[2].item()

    opter = ShapeOptConPts(cfd=cfd, ae=ae, cl_scale=scale_cl, cd_scale=scale_cd, vol_scale=scale_vol)
    xs = opter.solve_minCD_conCL(x0)
    # opter.draw_wings(x0)

    try:
        alfa_init, twist_init = opter.convert2xflr('init')
        alfa_opt, twist_opt = opter.convert2xflr('opt')
        # re_init, re_opt, c_init, c_opt = opter.compute_re()
        # opter.compare_airfoils(re_init, re_opt, twist_init, twist_opt, alfa_init, alfa_opt, c_init, c_opt)
        # opter.view_cl_icd_distribution()
    except OSError:
        pass



