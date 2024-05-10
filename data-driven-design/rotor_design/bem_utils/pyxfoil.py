import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil
from scipy import interpolate
from copy import copy
import aerosandbox


rho = 1.19252
nu = 1.81e-5
omega = 31.4  # rpm 300
radii = [0.61, 1.70, 4.77, 7.00]  # 7.32
dr = [0.545, 2.08, 2.65, 1.115+0.32]
# chord = [0.57, 0.60, 0.50, 0.44]
AR = 7.32/0.53
N_ALPHA = 181
# 0.61-->7.315, omega=31.4rad/s


def interpolate_(cald):
    ind = np.where(cald[:, -1] > -90)[0]
    # print('len ind: ', len(ind))
    if len(ind) < 4:
        return -1, -1, -1, False
    ind[-1] += 1
    flag = cald[:, -1] > -90
    cald_flag = cald[flag]
    cald_min = cald_flag[0]
    cald_max = cald_flag[-1]

    try:
        f_cl = interpolate.interp1d(cald_flag[:, 0], cald_flag[:, 1], kind='cubic')
        f_cd = interpolate.interp1d(cald_flag[:, 0], cald_flag[:, 2], kind='cubic')
    except ValueError:
        return -1, -1, -1, False
    cald[ind[0]:ind[-1], 1] = f_cl(cald[ind[0]:ind[-1], 0])
    cald[ind[0]:ind[-1], 2] = f_cd(cald[ind[0]:ind[-1], 0])
    cl_a_max = np.argmax(cald[:, 1])
    if cald[cl_a_max, 0] <= 0:
        return -1, -1, -1, False
    else:
        return cald[ind[0]:ind[-1]], cald_min, cald_max, True


def xfoil_cal(pts, radii_id):

    xf = XFoil()
    xf.print = False
    xf.airfoil = Airfoil(pts[:, 0], pts[:, 1])
    xf.repanel()
    v = omega*radii[radii_id]
    xf.Re = rho*v/nu
    xf.M = v / 343.2

    xf.max_iter = 100
    alphas_pos = np.linspace(0, 20, 41)
    alphas_neg = np.linspace(0, -20, 41)
    cald = np.zeros((81, 4))
    cald_pos = np.zeros((41, 4)) - 100
    cald_neg = np.zeros((41, 4)) - 100
    for i in range(len(alphas_pos)):
        xf.reset_bls()
        xf.repanel()
        alfa = alphas_pos[i]
        res_ = xf.a(alfa)
        cald_pos[i, 0] = alfa
        if np.isnan(sum(res_)):
            continue
        cald_pos[i, 1] = res_[0]
        cald_pos[i, 2] = res_[1]
        cald_pos[i, 3] = res_[2]

    for i in range(len(alphas_neg)):
        xf.reset_bls()
        xf.repanel()
        alfa = alphas_neg[i]
        res_ = xf.a(alfa)
        cald_neg[i, 0] = alfa
        if np.isnan(sum(res_)):
            continue
        cald_neg[i, 1] = res_[0]
        cald_neg[i, 2] = res_[1]
        cald_neg[i, 3] = res_[2]

    cald[:40] = cald_neg[::-1][:-1]
    cald[40:] = cald_pos
    print('....')
    cald, cald_min, cald_max, flag = interpolate_(cald)

    assert flag

    return cald

