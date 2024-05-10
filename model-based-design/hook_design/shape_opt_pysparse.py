import os
import numpy as np
import torch
from models import HookPhyNetFC
from models_ae import AERes
import scipy.io as scio
from pyoptsparse import SLSQP, Optimization, PSQP, IPOPT
from hook_genertor.hook_generator import HookGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prob', type=str, default='minEconV', help='the optimization',
                    choices=['minEconV', 'minVconE'])

# min SE con V
# MAX_ITER = 200
# AE_EPS = 0.01
# TOL = 1e-5


WEIGHT = 0.00000
con_scale = 0.5


# min V con SE
MAX_ITER = 200
AE_EPS = 0.01
TOL = 1e-5


class AENet:
    def __init__(self, ae, mu, sig):
        self.ae = ae
        self.mu = torch.from_numpy(mu).float()
        self.sig = torch.from_numpy(sig).float()

    def _only_guide_scale(self, x_torch):
        return x_torch[:, :11], x_torch[:, :66]

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def eval(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x).unsqueeze(0)
        recon = self.ae(xn)
        # xn1, xn2 = self._only_guide_scale(xn)
        # recon1, recon2 = self._only_guide_scale(recon)
        # return ((torch.abs(xn1 - recon1).mean()).item() + (torch.abs(xn2 - recon2).mean()).item()) / 2
        return (torch.abs(xn - recon).mean()).item()

    def grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch).unsqueeze(0)
        recon = self.ae(xn)
        y = torch.abs(xn - recon).mean()
        # xn1, xn2 = self._only_guide_scale(xn)
        # recon1, recon2 = self._only_guide_scale(recon)
        # y = (torch.abs(xn1 - recon1).mean() + torch.abs(xn2 - recon2).mean()) / 2
        y.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy

    def con_hess(self, x, v):
        return np.zeros((70, 70)) * v


class HookNet:
    def __init__(self, net, mu, sig):
        super(HookNet, self).__init__()
        self.net = net
        self.mu = torch.from_numpy(mu).float()
        self.sig = torch.from_numpy(sig).float()
        A = torch.eye(11)[:-1, :]
        A[:, 1:] -= torch.eye(10)
        self.A = A.float()

    def regular_term(self, x_torch):
        secs_pts_ = x_torch[11:66].reshape(11, 5)
        scales_ = x_torch[:11]
        scales_reg = torch.abs(self.A @ scales_).mean()
        secs_reg = torch.mean(torch.abs(self.A @ secs_pts_).sum(-1))
        return scales_reg + secs_reg

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def se_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch_n = self.normalized(x_torch)
        return self.net(x_torch_n.unsqueeze(0))[0, 0].item()

    def se_eval_regular(self, x, w):
        x_torch = torch.tensor(x, dtype=torch.float32)
        w_reg = w * self.regular_term(x_torch)
        x_torch_n = self.normalized(x_torch)
        return self.net(x_torch_n.unsqueeze(0))[0, 0].item() + w_reg.item()

    def vol_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch_n = self.normalized(x_torch)
        return self.net(x_torch_n.unsqueeze(0))[0, 1].item()

    def vol_eval_regular(self, x, w):
        x_torch = torch.tensor(x, dtype=torch.float32)
        w_reg = w * self.regular_term(x_torch)
        x_torch_n = self.normalized(x_torch)
        return self.net(x_torch_n.unsqueeze(0))[0, 1].item() + w_reg.item()

    def se_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        x_torch_n = self.normalized(x_torch)
        y_torch = self.net(x_torch_n.unsqueeze(0))
        y_torch[0, 0].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def se_grad_regular(self, x, w):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        w_reg = w * self.regular_term(x_torch)
        x_torch_n = self.normalized(x_torch)
        y_torch = self.net(x_torch_n.unsqueeze(0))
        yw_torch = y_torch[0, 0] + w_reg
        yw_torch.backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def vol_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        x_torch_n = self.normalized(x_torch)
        y_torch = self.net(x_torch_n.unsqueeze(0))
        y_torch[0, 1].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def vol_grad_regular(self, x, w):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        w_reg = w * self.regular_term(x_torch)
        x_torch_n = self.normalized(x_torch)
        y_torch = self.net(x_torch_n.unsqueeze(0))
        yw_torch = y_torch[0, 1] + w_reg
        yw_torch.backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def con_hess(self, x, v):
        return np.zeros((70, 70)) * v


class ShapeOptConPts:
    def __init__(self, hook_net, hook_ae, se_scale, vol_scale):
        self.hook_net = hook_net
        self.hook_ae = hook_ae
        self.se_scale = se_scale
        self.vol_scale = vol_scale

        I_74 = np.eye(74)
        I_scales = I_74[:11, :]
        I_sections = I_74[11:66, :]
        I_sections_1 = np.eye(55)[:5, :] @ I_sections
        B = np.array([
            [1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1]
        ])
        I_guide = I_74[66:, :]

        r = 0.025
        sqrt2 = 2 ** 0.5
        cLR = (1 - sqrt2) * r

        C = np.array([[1., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., -1., 1., 0., 0., 0., 0.],
                      [0., 0., 0., -1., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., -1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., -1., 0.]])

        self.A_scales = I_scales
        #                           s1     s2    s3     s4     s5      s6     s7     s8     s9    s10    s11
        self.scales_lb = np.array([0.02, 0.022, 0.022, 0.028, 0.026, 0.026, 0.028, 0.025, 0.025, 0.026, 0.03])
        self.scales_ub = np.array([0.03, 0.048, 0.048, 0.047, 0.048, 0.048, 0.047, 0.062, 0.063, 0.063, 0.06])
        self.A_sections1 = I_sections
        sec_lb_ = np.zeros((11, 5))
        sec_lb_[:, 0] = 0.12
        sec_lb_[:, -1] = 0.12
        self.sec_lb = sec_lb_.reshape(-1)
        self.A_sections2 = B @ I_sections_1
        self.A_guide1 = I_guide[[0, 1, 3, 4, 7]]
        self.g_lb = np.array([-r * sqrt2, r / sqrt2, r, r, 0.09])
        self.g_ub = np.array([-r / sqrt2, r * sqrt2, 2.5 * r, 2.5 * r, 0.14])
        self.A_guide2 = C @ I_guide
        self.g_lb2 = np.array([cLR, r / sqrt2, 0, cLR, 0, 0, 0])

        self.A_guide_scales = np.zeros(74)
        self.A_guide_scales[-2] = 1
        self.A_guide_scales[10] = 0.5

        self.A_guide8 = np.zeros(74)
        self.A_guide8[-1] = 1

        self.vol_hist = []
        self.se_hist = []
        self.ae_hist = []
        self.Niter = 0

    def save_xk_(self, xk, filename='xk'):
        scales = xk[:11]
        section_pts = xk[11:66].reshape(11, 5)
        guide_pts = xk[66:]
        hook = HookGenerator(order=0)
        hook.sec_pty = section_pts[np.newaxis]
        hook.sec_scales = scales[np.newaxis, :, np.newaxis]
        hook.guide_pts = guide_pts[np.newaxis]
        hook.get_guideline()
        hook.convert_for_matlab_(0, RESULT_DIR + '/%s' % filename)
        # hook.view_hook_i(0)
        # scio.savemat('%s/%s.mat' % (RESULT_DIR, filename),
        #              {'scales': scales, 'section_pts': section_pts, 'guide_pts': guide_pts})

    def print_init(self, xk):
        vol_k = self.hook_net.vol_eval(xk) * self.vol_scale
        se_k = self.hook_net.se_eval(xk) * self.se_scale
        ae_k = self.hook_ae.eval(xk)
        print('init volume: ', vol_k,
              'init strain energy: ', se_k,
              'init ae: ', ae_k)
        print('+++++++++++++++++++++++++++++++++++++++++')

    def minSE_conV(self, xdict):
        funcs = {}
        funcs['obj'] = self.hook_net.se_eval_regular(xdict['xvars'], WEIGHT)

        funcs['con1_1'] = [self.A_scales @ xdict['xvars']]
        funcs['con1_2'] = [self.A_sections1 @ xdict['xvars']]
        funcs['con1_3'] = [self.A_sections2 @ xdict['xvars']]
        funcs['con1_4'] = [self.A_guide1 @ xdict['xvars']]
        funcs['con1_5'] = [self.A_guide2 @ xdict['xvars']]
        funcs['con1_6'] = [self.A_guide_scales @ xdict['xvars']]
        funcs['con1_7'] = [self.A_guide8 @ xdict['xvars']]

        funcs['con2'] = [
            self.hook_net.vol_eval(xdict['xvars']),
            self.hook_ae.eval(xdict['xvars']),
        ]

        funcs['con3'] = [self.hook_net.se_eval(xdict['xvars'])]
        fail = False
        return funcs, fail

    def minSE_conV_sens(self, xdict, fdict):
        sens = {
            'obj': {'xvars': self.hook_net.se_grad_regular(xdict['xvars'], WEIGHT)},
            'con2': {'xvars': np.vstack(
                [
                    self.hook_net.vol_grad(xdict['xvars']),
                    self.hook_ae.grad(xdict['xvars']),
                ]
            )},
            'con3': {'xvars': self.hook_net.se_grad(xdict['xvars'])}
        }
        return sens

    def solve_minSE_conV(self, x0):
        self.print_init(x0)
        ae_eps = AE_EPS  # ae_mu - 3 * ae_sig
        g8_init = x0[-1]
        v0 = self.hook_net.vol_eval(x0)
        se0 = self.hook_net.se_eval(x0)

        optProb = Optimization('shape opt', self.minSE_conV)
        optProb.addVarGroup('xvars', 74, value=x0)
        optProb.addConGroup('con1_1', 11, lower=self.scales_lb, upper=self.scales_ub, linear=True,
                            jac={'xvars': self.A_scales})
        optProb.addConGroup('con1_2', 55, lower=self.sec_lb, upper=0.6, linear=True, jac={'xvars': self.A_sections1})
        optProb.addConGroup('con1_3', 4, lower=None, upper=0, linear=True, jac={'xvars': self.A_sections2})
        optProb.addConGroup('con1_4', 5, lower=self.g_lb, upper=self.g_ub, linear=True, jac={'xvars': self.A_guide1})
        optProb.addConGroup('con1_5', 7, lower=self.g_lb2, upper=None, linear=True, jac={'xvars': self.A_guide2})
        optProb.addConGroup('con1_6', 1, lower=0, upper=None, linear=True, jac={'xvars': self.A_guide_scales})
        # optProb.addConGroup('con1_7', 1, lower=g8_init, upper=g8_init, linear=True, jac={'xvars': self.A_guide8})

        optProb.addConGroup('con2', 2, lower=[None, None], upper=[con_scale * v0, ae_eps])
        # optProb.addConGroup('con3', 1, lower=[None], upper=[45./self.se_scale])
        optProb.addObj('obj')
        print(optProb)

        optOption = {'max_iter': MAX_ITER, 'tol': TOL}
        opt = IPOPT(options=optOption)

        # optOption = {'IPRINT': -1, 'MIT': MAX_ITER}
        # opt = PSQP(options=optOption)

        # optOption = {'IPRINT': -1, 'MAXIT': MAX_ITER}
        # opt = SLSQP(options=optOption)

        sol = opt(optProb, sens=self.minSE_conV_sens)
        print(sol)
        xs = np.array([v.value for v in sol.variables['xvars']])

        input('press anything to continue...')

        print('=====================================================================')
        print(self.se_scale, self.vol_scale)
        vol_0 = self.hook_net.vol_eval(x0) * self.vol_scale * 1000000
        se_0 = self.hook_net.se_eval(x0) * self.se_scale
        vol_1 = self.hook_net.vol_eval(xs) * self.vol_scale * 1000000
        se_1 = self.hook_net.se_eval(xs) * self.se_scale
        print('\ninit volume: ', vol_0, ' --> opt volume: ', vol_1,
              '\ninit strain energy: ', se_0, ' --> opt strain energy: ', se_1,
              '\ninit ae: ', self.hook_ae.eval(x0), ' --> opt ae: ', self.hook_ae.eval(xs),
              )
        self.save_xk_(x0, 'x_init')
        self.save_xk_(xs, 'x_opt')

    def minV_conSE(self, xdict):
        funcs = {}
        funcs['obj'] = self.hook_net.vol_eval_regular(xdict['xvars'], WEIGHT)

        funcs['con1_1'] = [self.A_scales @ xdict['xvars']]
        funcs['con1_2'] = [self.A_sections1 @ xdict['xvars']]
        funcs['con1_3'] = [self.A_sections2 @ xdict['xvars']]
        funcs['con1_4'] = [self.A_guide1 @ xdict['xvars']]
        funcs['con1_5'] = [self.A_guide2 @ xdict['xvars']]
        funcs['con1_6'] = [self.A_guide_scales @ xdict['xvars']]

        funcs['con2'] = [
            self.hook_net.se_eval(xdict['xvars']),
            self.hook_ae.eval(xdict['xvars']),
        ]
        fail = False
        return funcs, fail

    def minV_conSE_sens(self, xdict, fdict):
        sens = {
            'obj': {'xvars': self.hook_net.vol_grad_regular(xdict['xvars'], WEIGHT)},
            'con2': {'xvars': np.vstack(
                [
                    self.hook_net.se_grad(xdict['xvars']),
                    self.hook_ae.grad(xdict['xvars']),
                ]
            )}
        }
        return sens

    def solve_minV_conSE(self, x0):
        self.print_init(x0)
        ae_eps = AE_EPS  # ae_mu - 3 * ae_sig

        v0 = self.hook_net.vol_eval(x0)
        se0 = self.hook_net.se_eval(x0)

        optProb = Optimization('shape opt', self.minV_conSE)
        optProb.addVarGroup('xvars', 74, value=x0)
        optProb.addConGroup('con1_1', 11, lower=self.scales_lb, upper=self.scales_ub, linear=True,
                            jac={'xvars': self.A_scales})
        optProb.addConGroup('con1_2', 55, lower=self.sec_lb, upper=0.6, linear=True, jac={'xvars': self.A_sections1})
        optProb.addConGroup('con1_3', 4, lower=None, upper=0, linear=True, jac={'xvars': self.A_sections2})
        optProb.addConGroup('con1_4', 5, lower=self.g_lb, upper=self.g_ub, linear=True, jac={'xvars': self.A_guide1})
        optProb.addConGroup('con1_5', 7, lower=self.g_lb2, upper=None, linear=True, jac={'xvars': self.A_guide2})
        optProb.addConGroup('con1_6', 1, lower=0, upper=None, linear=True, jac={'xvars': self.A_guide_scales})

        optProb.addConGroup('con2', 2, lower=[None, None], upper=[con_scale * se0, ae_eps])
        optProb.addObj('obj')
        print(optProb)
        # optOption = {'IPRINT': -1, 'MIT': MAX_ITER}
        optOption = {'max_iter': MAX_ITER, 'tol': TOL}

        # opt = PSQP(options=optOption)
        opt = IPOPT(options=optOption)
        sol = opt(optProb, sens=self.minV_conSE_sens)
        print(sol)
        xs = np.array([v.value for v in sol.variables['xvars']])

        input('press anything to continue...')

        print('=====================================================================')
        vol_0 = self.hook_net.vol_eval(x0) * self.vol_scale * 1000000
        se_0 = self.hook_net.se_eval(x0) * self.se_scale
        vol_1 = self.hook_net.vol_eval(xs) * self.vol_scale * 1000000
        se_1 = self.hook_net.se_eval(xs) * self.se_scale
        print('\ninit volume: ', vol_0, ' --> opt volume: ', vol_1,
              '\ninit strain energy: ', se_0, ' --> opt strain energy: ', se_1,
              '\ninit ae: ', self.hook_ae.eval(x0), ' --> opt ae: ', self.hook_ae.eval(xs),
              )
        self.save_xk_(x0, 'x_init')
        self.save_xk_(xs, 'x_opt')


if __name__ == '__main__':
    args = parser.parse_args()
    def _init_fn(worker_id):
        np.random.seed(1 + worker_id)


    prefix = 'al-semi'
    player = 'teacher'
    RESULT_DIR = 'optimized_results/HOOK-AL_SEMI_GEO_%s_opt_%s'
    RESULT_DIR = RESULT_DIR % (prefix, args.prob)

    """
    minSE conV:
    init volume:  tensor(187.5531)  --> opt volume:  tensor(93.8605)  [43.3729]
    init strain energy:  tensor(129.5951)  --> opt strain energy:  tensor(41.0309) [95.8597]
    init ae:  0.13364268839359283  --> opt ae:  0.01277248840779066

    minVconSE:
    init volume:  tensor(187.5531)  --> opt volume:  tensor(70.4850) [67.3347]
    init strain energy:  tensor(129.5951)  --> opt strain energy:  tensor(64.8172) [64.9608]
    init ae:  0.13364268839359283  --> opt ae:  0.015201356261968613
    """

    try:
        os.mkdir(RESULT_DIR)
    except OSError:
        pass

    data_train = np.load('data/data_train.npy', allow_pickle=True).item()
    # data_val = np.load('data/data_val.npy', allow_pickle=True).item()
    # x0 = data_train['params'][817]
    x0 = np.load('optimized_results/x_init.npy')

    hook_net_ = HookPhyNetFC(in_dim=74)
    check_net = torch.load('mean-teacher-al/results-%s/net_best_%s.tar' % (prefix, player))
    mu = check_net['option']['mu_paras']
    sig = check_net['option']['sig_paras']
    hook_net_.load_state_dict(check_net[player])
    hook_net_.cpu()
    hook_net_.eval()
    hook_net = HookNet(net=hook_net_, mu=mu, sig=sig)

    ID = 68
    ae_net_ = AERes(embed_dim=ID,
                    ndfs=[96, 96, 96, 96],
                    ngfs=[96, 96, 96, 96])
    check_ae = torch.load('shape-anomaly-detection/results_aeID=%d/ae_best_zeros.tar' % ID)
    ae_net_.load_state_dict(check_ae['net_state_dict'])
    ae_net_.cpu()
    ae_net_.eval()

    ae_net = AENet(ae=ae_net_, mu=check_ae['option']['mu'], sig=check_ae['option']['sig'])

    scale_se = check_net['scales'][0]  # check_net['option']['se_var']
    scale_vol = check_net['scales'][1]  # check_net['option']['vol_var']
    print(scale_se, scale_vol)

    opter = ShapeOptConPts(hook_net=hook_net, hook_ae=ae_net, se_scale=scale_se, vol_scale=scale_vol)
    if RESULT_DIR.endswith('opt_minV_conSE'):
        opter.solve_minV_conSE(x0)
    else:
        opter.solve_minSE_conV(x0)
