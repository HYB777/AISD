import torch
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.optimize as opt
import scipy.io as scio


def _basis_(i, d, x, knots):

    if d == 0:
        return ((knots[i] <= x < knots[i + 1]) or (x == 1 and knots[i + 1] == 1.0)) * 1.0
    val = 0
    if knots[i + d] != knots[i]:
        val += ((x - knots[i]) / (knots[i + d] - knots[i])) * _basis_(i, d - 1, x, knots)
    if knots[i + d + 1] != knots[i + 1]:
        val += ((knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1])) * _basis_(i + 1, d - 1, x, knots)
    return val


def get_knots(degree, n_controls):
    knots = np.zeros(n_controls + degree + 1)
    for i in range(n_controls + degree + 1):
        if i <= degree:
            knots[i] = 0
        elif degree + 1 <= i <= n_controls - 1:
            knots[i] = (i - degree) / (n_controls + 1 - degree)
        else:
            knots[i] = 1
    return knots


def get_M(x_upper, x_lower, n_controls, knots, degree=3):

    M_upper = sparse.dok_matrix((len(x_upper), n_controls - 2))
    M_lower = sparse.dok_matrix((len(x_lower), n_controls - 2))
    for i in range(len(x_upper)):
        x = x_upper[i]
        for j in range(1, n_controls - 1):
            Nji = _basis_(j, degree, x, knots)
            if Nji != 0:
                M_upper[i, j - 1] += Nji

    for i in range(len(x_lower)):
        x = x_lower[i]
        for j in range(1, n_controls - 1):
            Nji = _basis_(j, degree, x, knots)
            if Nji != 0:
                M_lower[i, j - 1] += Nji

    return M_upper.tocsc(), M_lower.tocsc()


def get_N(n_controls, knots, T, degree=3):
    N = sparse.dok_matrix((T, n_controls))
    t = np.linspace(0, 1, T)
    for i in range(T):
        ti = t[i]
        for j in range(n_controls):
            Nji = _basis_(j, degree, ti, knots)
            if Nji != 0:
                N[i, j] += Nji
    return N.tocsc()


def eval(u, knots, controls, degree=3):
    val = 0
    for j in range(len(controls)):
        val += _basis_(j, degree, u, knots) * controls[j]
    return val



