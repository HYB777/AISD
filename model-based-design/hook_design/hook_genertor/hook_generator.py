import numpy as np

from .find_simplex_vertex import init_hooker
from .bspline_utils import *
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


class HookGenerator:
    def __init__(self, order=1, filename='.'):
        super(HookGenerator, self).__init__()
        self.filename = filename
        self.order = order
        self.SEC_PTY_LB = np.array([0.12, 0, 0, 0, 0.12])
        self.SEC_PTY_UB = np.array([0.6, 0.6, 0.6, 0.6, 0.6])

        self.SEC_SCALES_LB = np.array([0.02, 0.022, 0.022, 0.028, 0.026, 0.026, 0.028, 0.025, 0.025, 0.026, 0.03])
        self.SEC_SCALES_UB = np.array([0.03, 0.048, 0.048, 0.047, 0.048, 0.048, 0.047, 0.062, 0.063, 0.063, 0.06])

        self.R = 0.025
        self.SQRT2 = 2**0.5
        self.G12_PTS = np.array([
            [-self.R / self.SQRT2, self.R / self.SQRT2],
            [-self.R / self.SQRT2, self.SQRT2 * self.R],
            [-self.SQRT2 * self.R, self.SQRT2 * self.R],
            [-self.SQRT2 * self.R, self.R],
            [(1 - 3 / self.SQRT2) * self.R, self.R / self.SQRT2]
        ])
        self.G45_PTS = np.array([
            [self.R, self.R],
            [self.R, 2.5 * self.R],
            [2 * self.R, 2.5 * self.R],
            [2 * self.R, (3 - self.SQRT2) * self.R],
            [self.SQRT2 * self.R, self.R]
        ])

        filter_ = np.eye(11)
        filter_[1:, :-1] += np.eye(10)
        filter_[:-1, 1:] += np.eye(10)
        filter_ /= filter_.sum(1).reshape(-1, 1)
        # filter_[1:-1, :-2] += np.eye(9)
        # filter_[1:-1, 2:] += np.eye(9)
        # filter_[1:-1] /= 3.
        self.filter_ = np.eye(11)
        for i in range(order):
            self.filter_ = self.filter_ @ filter_

        thetas = np.linspace(-3, -1, 100) / 4 * np.pi
        self.ARC = [self.R * np.cos(thetas), self.R * np.sin(thetas) + self.R]
        self.LPTS = [-self.R / self.SQRT2, (1 - 1 / self.SQRT2) * self.R]
        self.RPTS = [ self.R / self.SQRT2, (1 - 1 / self.SQRT2) * self.R]

        x_controls = np.array([0, 0, 0.25, 0.5, 0.75, 1, 1])
        self.Nsec = self.get_N_mat_(n_controls=7, t=50, prev_fix='Nsec')
        self.x_hat_sec = self.Nsec @ x_controls
        self.Nguide1 = self.get_N_mat_(n_controls=3, t=100, degree=2, prev_fix='Nguide1')
        self.Nguide2 = self.get_N_mat_(n_controls=7, t=500, prev_fix='Nguide2')
        self.INTERPOLATE_G58 = np.array([
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75]
        ]).transpose()

        self.x1_hat = None
        self.y1_hat = None
        self.x2_hat = None
        self.y2_hat = None
        self.sec_pty = None
        self.sec_scales = None
        self.guide_pts = None

        self.guideline_ptX1 = None
        self.guideline_ptY1 = None
        self.guideline_ptX2 = None
        self.guideline_ptY2 = None

        self.SEC_POLYGON_MODES = init_hooker()

    def __len__(self):
        return len(self.sec_pty)

    @staticmethod
    def curvature_max_(curveX, curveY):
        dx = (curveX[:, 2:] - curveX[:, :-2]) / 2.
        dy = (curveY[:, 2:] - curveY[:, :-2]) / 2.
        ddx = curveX[:, 2:] + curveX[:, :-2] - 2 * curveX[:, 1:-1]
        ddy = curveY[:, 2:] + curveY[:, :-2] - 2 * curveY[:, 1:-1]
        curvature_ = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
        return curvature_.max(1)

    @staticmethod
    def rot_y_n_(xyz, n):
        cos_theta = n[1]
        sin_theta = -n[0]
        rot = np.array([[cos_theta, 0, -sin_theta],
                        [0, 1, 0],
                        [sin_theta, 0, cos_theta]])
        return rot @ xyz

    @staticmethod
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    @staticmethod
    def get_locates_(curveX, curveY, n_locates):
        ds = ((curveX[1:] - curveX[:-1])**2 + (curveY[1:] - curveY[:-1])**2)**0.5
        s = np.sum(ds)
        per_s = s / (n_locates + 1)
        s_ = 0
        cnt = 0
        locates_ = np.zeros((n_locates, 2))
        normals_ = np.zeros((n_locates, 2))
        for i in range(len(ds)):
            s_ += ds[i]
            if s_ >= per_s * (cnt + 1):
                locates_[cnt] = [curveX[i + 1], curveY[i + 1]]
                tangent_ = np.array([curveX[i + 2] - curveX[i], curveY[i + 2] - curveY[i]])
                tangent_ /= np.linalg.norm(tangent_)
                normals_[cnt] = [tangent_[1], -tangent_[0]]
                cnt += 1
            if cnt == n_locates:
                break
        return locates_, normals_

    @staticmethod
    def get_N_mat_(n_controls, t=50, degree=3, prev_fix='N', direct='./hook_genertor/config'):
        try:
            N = np.load('%s/%s_c%d_d%d_t%d.npy' % (direct, prev_fix, n_controls, degree, t), allow_pickle=True).item()
        except FileNotFoundError:
            knots = get_knots(degree=degree, n_controls=n_controls)
            N = get_N(n_controls=n_controls, knots=knots, T=t, degree=degree)
            np.save('%s/%s_c%d_d%d_t%d.npy' % (direct, prev_fix, n_controls, degree, t), N)
        return N

    @staticmethod
    def rand_gen_(lb, ub, ratio, num):
        len_ = ub - lb
        mid = (lb + ub) / 2
        lb_ = mid - len_ * ratio / 2
        ub_ = mid + len_ * ratio / 2
        w = np.random.random(num)
        return w * lb_ + (1 - w) * ub_

    def generate_sec_pty(self, num):
        # sec_pty = np.random.random((num, 11, 5)) * (self.SEC_PTY_UB - self.SEC_PTY_LB) + self.SEC_PTY_LB
        n_mode = num // 20
        n_mix = num - 16 * n_mode
        sec_pty_modes = []
        for mode_i in self.SEC_POLYGON_MODES:
            n_vert = len(mode_i)
            wi_ = np.random.random((11 * n_mode, n_vert))
            wi = wi_ / wi_.sum(1).reshape(-1, 1)
            sec_pty_modes.append(wi @ mode_i)

        sec_pty_mixes = []
        n_mix_per_mode = [n_mix // 16] * 15
        n_mix_per_mode.append(n_mix - 15 * n_mix // 16)
        for i in range(16):
            mode_i = self.SEC_POLYGON_MODES[i]
            n_vert = len(mode_i)
            wi_ = np.random.random((11 * n_mix_per_mode[i], n_vert))
            wi = wi_ / wi_.sum(1).reshape(-1, 1)
            sec_pty_mixes.append(wi @ mode_i)
        sec_pty_mixes = np.vstack(sec_pty_mixes)
        np.random.shuffle(sec_pty_mixes)
        sec_pty_modes = np.vstack(sec_pty_modes)
        sec_pty = np.vstack([sec_pty_modes, sec_pty_mixes]).reshape((num, 11, 5))
        return sec_pty

    def generate_sec_scale(self, num):
        sec_scales = np.random.random((num, 11)) * (self.SEC_SCALES_UB - self.SEC_SCALES_LB) + self.SEC_SCALES_LB
        return sec_scales

    def generate_guideline(self, num, s11, ratio=0.8):
        w12_ = np.random.random((num, 5))
        w12 = w12_ / w12_.sum(1).reshape(-1, 1)
        g12 = w12 @ self.G12_PTS

        w45_ = np.random.random((num, 5))
        w45 = w45_ / w45_.sum(1).reshape(-1, 1)
        g45 = w45 @ self.G45_PTS

        g3 = self.rand_gen_(self.R / self.SQRT2, g45[:, 0], ratio, num).reshape(-1, 1)
        g6 = self.rand_gen_(0, g45[:, 0], ratio, num).reshape(-1, 1)
        g7 = self.rand_gen_(-0.5 * s11.reshape(-1), 0, ratio, num).reshape(-1, 1)
        g8 = self.rand_gen_(0.09, 0.14, 1, num).reshape(-1, 1)

        g = np.hstack([g12, g3, g45, g6, g7, g8])
        return g

    def generate_hook(self, num):
        self.sec_pty = self.filter_ @ self.generate_sec_pty(num)
        self.sec_scales = self.filter_ @ self.generate_sec_scale(num)[:, :, np.newaxis]
        self.guide_pts = self.generate_guideline(num, self.sec_scales[:, -1])

    def get_sec_curve(self, sec_pty_):
        y_controls = np.zeros((11, 7))
        y_controls[:, 1:-1] = sec_pty_
        y_hat = y_controls @ self.Nsec.transpose()
        return y_hat

    def get_guideline(self):
        s11 = self.sec_scales[:, -1].reshape(-1)
        ONES = np.ones_like(s11)
        ZEROS = np.zeros_like(s11)
        self.guideline_ptX1 = np.column_stack([self.guide_pts[:, 0], self.guide_pts[:, 0], ONES * self.LPTS[0]])
        self.guideline_ptY1 = np.column_stack([self.guide_pts[:, 1], -self.guide_pts[:, 0] + (1 - self.SQRT2) * self.R, ONES * self.LPTS[1]])
        self.x1_hat = (self.Nguide1 @ self.guideline_ptX1.transpose()).transpose()
        self.y1_hat = (self.Nguide1 @ self.guideline_ptY1.transpose()).transpose()

        g8x = -0.5 * s11
        self.guideline_ptX2 = np.column_stack([self.RPTS[0] * ONES, self.guide_pts[:, 2], self.guide_pts[:, 3], self.guide_pts[:, 5], self.guide_pts[:, 6], g8x, g8x])
        self.guideline_ptY2 = np.column_stack([self.RPTS[1] * ONES, self.guide_pts[:, 2] + (1 - self.SQRT2) * self.R, self.guide_pts[:, 4], ZEROS, ZEROS, ZEROS, self.guide_pts[:,  -1]])
        self.guideline_ptY2[:, 3:-1] = np.column_stack([self.guide_pts[:, 4], self.guide_pts[:,  -1]]) @ self.INTERPOLATE_G58
        self.x2_hat = (self.Nguide2 @ self.guideline_ptX2.transpose()).transpose()
        self.y2_hat = (self.Nguide2 @ self.guideline_ptY2.transpose()).transpose()

    def view_hook_i(self, i):
        sec_curves = self.get_sec_curve(self.sec_pty[i])
        locates1, normals1 = self.get_locates_(self.x1_hat[i], self.y1_hat[i], 1)
        far_ind = self.x2_hat[i].argmax()
        locates2_1, normals2_1 = self.get_locates_(self.x2_hat[i, :far_ind + 1], self.y2_hat[i, :far_ind + 1], 1)
        locates2_2, normals2_2 = self.get_locates_(self.x2_hat[i, far_ind:], self.y2_hat[i, far_ind:], 3)
        locates = np.vstack([
            np.array([[self.guide_pts[i, 0], self.guide_pts[i, 1]]]),
            locates1,
            np.array([[-self.R / self.SQRT2, (1 - 1 / self.SQRT2) * self.R]]),
            np.array([[0, 0]]),
            np.array([[self.R / self.SQRT2, (1 - 1 / self.SQRT2) * self.R]]),
            locates2_1,
            np.array([[self.x2_hat[i, far_ind], self.y2_hat[i, far_ind]]]),
            locates2_2,
            np.array([[-0.5 * self.sec_scales[i, -1].reshape(-1), self.guide_pts[i, -1]]]),
        ])
        normals = np.vstack([
            np.array([[-1, 0]]),
            normals1,
            np.array([[-1 / self.SQRT2, - 1 / self.SQRT2]]),
            np.array([[0, -1]]),
            np.array([[1 / self.SQRT2, - 1 / self.SQRT2]]),
            normals2_1,
            np.array([[1, 0]]),
            np.array([[1, 0]]),
            np.array([[1, 0]]),
            np.array([[1, 0]]),
            np.array([[1, 0]]),
        ])

        ax = Axes3D(plt.figure())
        ax.plot(self.x1_hat[i], np.zeros_like(self.x1_hat[i]), self.y1_hat[i], 'r-')
        ax.plot(self.ARC[0], np.zeros_like(self.ARC[0]), self.ARC[1], 'r-')
        ax.plot(self.x2_hat[i], np.zeros_like(self.x2_hat[i]), self.y2_hat[i], 'r-')
        ax.plot(self.guideline_ptX1[i], np.zeros_like(self.guideline_ptX1[i]), self.guideline_ptY1[i], 'go-')
        ax.plot(self.guideline_ptX2[i], np.zeros_like(self.guideline_ptX2[i]), self.guideline_ptY2[i], 'go-')
        for j in range(len(sec_curves)):
            x_ = np.zeros_like(self.x_hat_sec)
            y_ = self.x_hat_sec * self.sec_scales[i, j]
            z_ = sec_curves[j] * self.sec_scales[i, j]
            xyz_ = self.rot_y_n_(np.vstack([x_, z_, y_]), normals[j])
            ax.plot(xyz_[0] + locates[j, 0], xyz_[1], xyz_[2] + locates[j, 1], 'b')

        self.set_axes_equal(ax)
        plt.show()

    def curvature_list(self):
        return np.maximum(self.curvature_max_(self.x1_hat, self.y1_hat), self.curvature_max_(self.x2_hat, self.y2_hat))

    def continuous_list(self):
        sec_pts_scales = self.sec_scales * self.sec_pty
        return np.abs(sec_pts_scales[:, 1:] - sec_pts_scales[:, :-1]).reshape(-1, 50).sum(1)

    def screening(self, idx):
        self.sec_pty = self.sec_pty[idx]
        self.sec_scales = self.sec_scales[idx]
        self.guide_pts = self.guide_pts[idx]
        self.x1_hat, self.y1_hat = self.x1_hat[idx], self.y1_hat[idx]
        self.x2_hat, self.y2_hat = self.x2_hat[idx], self.y2_hat[idx]

        self.guideline_ptX1, self.guideline_ptY1 = self.guideline_ptX1[idx], self.guideline_ptY1[idx]
        self.guideline_ptX2, self.guideline_ptY2 = self.guideline_ptX2[idx], self.guideline_ptY2[idx]

        params = {
            'sec_pty': self.sec_pty,
            'sec_scales': self.sec_scales,
            'guide_pts': self.guide_pts,
        }
        np.save('params_ord%d.npy' % self.order, params)

    def convert_for_matlab_(self, i, filename=None):
        if self.x2_hat is None:
            self.get_guideline()
        sec_pty = self.sec_pty[i]
        sec_scales = self.sec_scales[i].reshape(-1)
        guide_pts = self.guide_pts[i]
        x1_hat, y1_hat = self.x1_hat[i], self.y1_hat[i]
        x2_hat, y2_hat = self.x2_hat[i], self.y2_hat[i]
        guideline_curve = np.vstack([np.column_stack([x1_hat[:-1], y1_hat[:-1]]),
                                     np.column_stack([self.ARC[0], self.ARC[1]]),
                                     np.column_stack([x2_hat[1:], y2_hat[1:]])])
        sec_curves_y = self.get_sec_curve(sec_pty)
        sec_curves_x = np.repeat(self.x_hat_sec[np.newaxis], 11, axis=0)
        sec_curves = np.concatenate([sec_curves_x[:, np.newaxis], sec_curves_y[:, np.newaxis]], axis=1).transpose((2, 1, 0))
        locates1, normals1 = self.get_locates_(x1_hat, y1_hat, 1)
        far_ind = x2_hat.argmax()
        locates2_1, normals2_1 = self.get_locates_(x2_hat[:far_ind + 1], y2_hat[:far_ind + 1], 1)
        locates2_2, normals2_2 = self.get_locates_(x2_hat[far_ind:], y2_hat[far_ind:], 3)
        locates = np.vstack([
            np.array([[guide_pts[0], guide_pts[1]]]),
            locates1,
            np.array([[-self.R / self.SQRT2, (1 - 1 / self.SQRT2) * self.R]]),
            np.array([[0, 0]]),
            np.array([[self.R / self.SQRT2, (1 - 1 / self.SQRT2) * self.R]]),
            locates2_1,
            np.array([[x2_hat[far_ind], y2_hat[far_ind]]]),
            locates2_2,
            np.array([[-0.5 * sec_scales[-1], guide_pts[-1]]]),
        ])
        normals = np.vstack([
            np.array([[-1, 0]]),
            normals1,
            np.array([[-1 / self.SQRT2, - 1 / self.SQRT2]]),
            np.array([[0, -1]]),
            np.array([[1 / self.SQRT2, - 1 / self.SQRT2]]),
            normals2_1,
            np.array([[1, 0]]),
            np.array([[1, 0]]),
            np.array([[1, 0]]),
            np.array([[1, 0]]),
            np.array([[1, 0]]),
        ])
        tangents = (np.array([
            [0, -1],
            [1, 0]
        ]) @ normals.transpose()).transpose()
        input_matlab = {
            'guideline_curve': guideline_curve,
            'sec_curves': sec_curves,
            'locates': locates,
            'tangents': tangents,
            'sec_scales': sec_scales
        }
        if filename is None:
            scio.savemat('%s/hook_dataset/hook_order%d_%d.mat' % (self.filename, self.order, i), input_matlab)
        else:
            scio.savemat('%s.mat' % filename, input_matlab)

    def generate_init(self):
        self.sec_pty = np.zeros((1, 11, 5)) + np.array([[0.12, 0.6, 0.6, 0.6, 0.12]])
        self.sec_scales = 0.5 * (self.SEC_SCALES_LB + self.SEC_SCALES_UB).reshape((1, 11, 1))
        s11 = self.sec_scales[0, -1, 0]
        g12 = self.G12_PTS.mean(0)
        g45 = self.G45_PTS.mean(0)
        g3 = np.array([g45[0] + self.R / self.SQRT2]).reshape(-1) * 0.5
        g6 = np.array([g45[0] / 2]).reshape(-1)
        g7 = -0.25 * s11.reshape(-1).reshape(-1)
        g8 = np.array([0.115]).reshape(-1)

        self.guide_pts = np.hstack([g12, g3, g45, g6, g7, g8]).reshape(1, -1)
        x_init = np.hstack([self.sec_scales.reshape(-1), self.sec_pty.reshape(-1), self.guide_pts.reshape(-1)])
        np.save('../optimized_results/x_init.npy', x_init)


if __name__ == '__main__':

    # hooker1 = HookGenerator(order=1)
    # hooker1.generate_init()
    # hooker1.get_guideline()
    # hooker1.view_hook_i(0)

    NUMBER = 6_0000
    #
    hooker1 = HookGenerator(order=1)
    # hooker2 = HookGenerator(order=2)
    hooker1.generate_hook(NUMBER * 3)
    # hooker2.generate_hook(NUMBER * 2)
    hooker1.get_guideline()
    # hooker2.get_guideline()
    # curvature_list1 = hooker1.curvature_list()
    # continuous_list1 = hooker1.continuous_list()
    # curvature_list2 = hooker2.curvature_list()
    # continuous_list2 = hooker2.continuous_list()
    # mu1, sig1 = continuous_list1.mean(), continuous_list1.std()
    # mu2, sig2 = continuous_list2.mean(), continuous_list2.std()
    #
    # g8g5_1 = 0.75 * hooker1.guide_pts[:, -1] + 0.25 * hooker1.guide_pts[:, 4] > hooker1.guide_pts[:, 1]
    # g8g5_2 = 0.75 * hooker2.guide_pts[:, -1] + 0.25 * hooker2.guide_pts[:, 4] > hooker2.guide_pts[:, 1]
    #
    # idx1 = (continuous_list1 <= mu1 + sig1) * (curvature_list1 < 200) * g8g5_1
    # idx2 = (continuous_list2 <= mu2 + sig2) * (curvature_list2 < 200) * g8g5_2
    #
    # print(np.sum(idx1), np.sum(idx2), np.prod(g8g5_1), np.prod(g8g5_2))
    # hooker1.screening(idx1)
    # hooker2.screening(idx2)
    #
    # for i in tqdm(range(len(hooker1))):
    #     hooker1.convert_for_matlab_(i)
    #
    # for i in tqdm(range(len(hooker2))):
    #     hooker2.convert_for_matlab_(i)
    # print('...')
