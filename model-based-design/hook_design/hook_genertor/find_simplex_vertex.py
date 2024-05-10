import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


def init_hooker():
    A = np.array([
        [1, -1, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [0, 0, 0, -1, 1, 0]])

    A_list = []
    bin_ = [-1, 1]
    for bi in bin_:
        for bj in bin_:
            for bk in bin_:
                for bs in bin_:
                    Aijks = np.copy(A)
                    Aijks[0] *= bi
                    Aijks[1] *= bj
                    Aijks[2] *= bk
                    Aijks[3] *= bs
                    A_list.append(Aijks)

    B = np.hstack([np.eye(5), -np.ones((5, 1)) * 0.6])
    C = np.hstack([-np.eye(5), np.zeros((5, 1))])
    C[0, -1] = 0.12
    C[-1, -1] = 0.12

    def find_(A_, B_, C_):
        halfspaces = np.vstack([A_, B_, C_])
        norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                                 (halfspaces.shape[0], 1))
        c = np.zeros((halfspaces.shape[1],))
        c[-1] = -1
        A = np.hstack((halfspaces[:, :-1], norm_vector))
        b = - halfspaces[:, -1:]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
        x = res.x[:-1]
        # y = res.x[-1]
        hs = HalfspaceIntersection(halfspaces, x)
        return hs.intersections

    return [find_(A_, B, C) for A_ in A_list]
