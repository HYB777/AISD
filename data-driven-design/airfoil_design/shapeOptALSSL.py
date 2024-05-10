import os
from nets.nets import *
from nets.ae_nets import *
from bspline_utils import BsplineArea
from pyoptsparse import Optimization, PSQP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ma', type=int, default=734, help='mach number * 1000', choices=[150, 734])
args = parser.parse_args()

MACH = args.ma
ALFA_MAX = 10 if MACH == 150 else 3
MAX_ITER = 200
filename = 'naca2412'
CL_TAG = 0.8
AE_EPS = 1e-3
RESULT_DIR = 'optimize_results/%s_minCD_conCL_ALSSL_ma%d' % (filename, MACH)

try:
    os.mkdir(RESULT_DIR)
except OSError:
    pass


class AENet:
    def __init__(self, ae, mu, sig):
        self.ae = ae
        self.mu = mu
        self.sig = sig

    def normalized(self, x):
        return (x - self.mu) / self.sig

    def eval(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        xn = self.normalized(x[:-1].reshape(1, 2, -1))
        recon = self.ae(xn)
        return (torch.abs(xn - recon).mean()).detach().numpy().reshape(-1)

    def grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        xn = self.normalized(x_torch[:-1].reshape(1, 2, -1))
        recon = self.ae(xn)
        y = torch.abs(xn - recon).mean()
        y.backward()
        gxy = x_torch.grad.data.detach().numpy().reshape(1, -1)
        return gxy

    def con_hess(self, x, v):
        return np.zeros((65, 65)) * v


class Areaer:
    def __init__(self, bs):
        super(Areaer, self).__init__()
        self.bs = bs

    def eval(self, x):
        return self.bs.eval(x[:-1])

    def grad(self, x):
        return np.hstack([self.bs.grad(x[:-1]), 0])

    def hess(self, x):
        h = self.bs.hess(x)
        H = np.zeros((65, 65))
        H[:32, :32] = h[1:-1, 1:-1]
        H[32:-1, 32:-1] = -h[1:-1, 1:-1]
        return H

    def con_hess(self, x, v):
        h = self.bs.hess(x)
        H = np.zeros((65, 65))
        H[:32, :32] = h[1:-1, 1:-1]
        H[32:-1, 32:-1] = -h[1:-1, 1:-1]
        return H * v


class CFDNet:
    def __init__(self, net):
        super(CFDNet, self).__init__()
        self.net = net

    def eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        cl, cd = self.net(x_torch[:-1].reshape(1, 2, -1), x_torch[-1])
        return cl.detach().numpy().reshape(-1), cd.detach().numpy().reshape(-1)

    def cl_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        return self.net(x_torch[:-1].reshape(1, 2, -1), x_torch[-1])[0].detach().numpy().reshape(-1)

    def cd_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        return self.net(x_torch[:-1].reshape(1, 2, -1), x_torch[-1])[1].detach().numpy().reshape(-1)

    def clcd_eval(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        cl, cd, _, _ = self.net(x_torch[:-1].reshape(1, 2, -1), x_torch[-1])
        return cl.detach().numpy().reshape(-1) / cd.detach().numpy().reshape(-1)

    def cl_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:-1].reshape(1, 2, -1), x_torch[-1])
        y_torch[0].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def cd_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:-1].reshape(1, 2, -1), x_torch[-1])
        y_torch[1].backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def clcd_grad(self, x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        x_torch.requires_grad_(True)

        y_torch = self.net(x_torch[:-1].reshape(1, 2, -1), x_torch[-1])
        clcd = y_torch[0] / y_torch[1]
        clcd.backward()

        gxy = x_torch.grad.data.numpy().reshape(1, -1)

        return gxy

    def hess(self, x):
        return np.zeros((65, 65))

    def con_hess(self, x, v):
        return np.zeros((65, 65)) * v

    def clcd_hess(self, x, alfa):
        return np.zeros((65, 65))

    def con_clcd_hess(self, x, v, alfa):
        return np.zeros((65, 65)) * v


class ShapeOptConPts:
    def __init__(self, cfd, ae, area, cl_scale, cd_scale):
        self.cfd = cfd
        self.ae = ae
        self.area = area
        self.cl_scale = cl_scale
        self.cd_scale = cd_scale

        I = np.eye(32)
        self.A = np.zeros((32, 64 + 1))
        self.A[:, :-1] = np.hstack([I, -I])
        self.B = np.zeros((32, 64 + 1))
        self.B[:, :32] = I
        self.c1 = np.zeros(64 + 1)
        self.c2 = np.zeros(64 + 1)
        self.c3 = np.zeros(64 + 1)
        self.c1[0] = 1
        self.c2[32] = 1
        self.c3[-1] = 1

        self.area_hist = []
        self.cl_hist = []
        self.cd_hist = []
        self.clcd_hist = []
        self.ae_hist = []
        self.Niter = 0
        self.alfa = None

    def print_init(self, xk):
        area_k = self.area.eval(xk)
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
        funcs['con1'] = [
            self.A @ xdict['xvars'],
        ]
        funcs['con2'] = [
            self.cfd.cl_eval(xdict['xvars']).item(),
            self.ae.eval(xdict['xvars']).item(),
            self.area.eval(xdict['xvars']).item(),
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
            # 'con1': {'xvars': [
            #     self.A,
            # ]},
            'con2': {'xvars': np.vstack(
                [
                    # self.A,
                    self.cfd.cl_grad(xdict['xvars']),
                    self.ae.grad(xdict['xvars']),
                    self.area.grad(xdict['xvars']),
                ]
            )}
        }
        return sens

    def solve_minCD_conCL(self, x0):
        self.alfa = alfa

        self.print_init(x0)
        ae_eps = AE_EPS  # ae_mu - 3 * ae_sig

        A0 = self.area.eval(x0)
        eA = 0.1  # 0.1, 1
        cl0 = self.cfd.cl_eval(x0).item() * self.cl_scale
        thick = self.A@x0

        optProb = Optimization('shape opt', self.minCD_conCL)
        optProb.addVarGroup('xvars', 64 + 1, value=x0)
        optProb.addConGroup('con1', 32, lower=thick * 0.0, linear=True, jac={'xvars': self.A})
        optProb.addConGroup('con2', 3, lower=[CL_TAG/self.cl_scale, None, eA*A0], upper=[None, ae_eps, None])
        optProb.addConGroup('con3', 32, lower=np.zeros(32), linear=True, jac={'xvars': self.B})
        optProb.addConGroup('con4', 1, upper=0, linear=True, jac={'xvars': self.c2})
        optProb.addConGroup('con5', 1, lower=0.5, upper=ALFA_MAX-0.5, linear=True, jac={'xvars': self.c3})
        optProb.addObj('obj')
        print(optProb)
        optOption = {'IPRINT': -1, 'MIT': MAX_ITER}

        opt = PSQP(options=optOption)
        sol = opt(optProb, sens=self.minCD_conCL_sens)
        print(sol)
        xs = np.array([v.value for v in sol.variables['xvars']])
        input('press anything to continue...')

        print('=====================================================================, MACH: %f', MACH / 1000.)
        cls0 = self.cfd.cl_eval(x0) * self.cl_scale
        cds0 = self.cfd.cd_eval(x0) * self.cd_scale
        cls1 = self.cfd.cl_eval(xs) * self.cl_scale
        cds1 = self.cfd.cd_eval(xs) * self.cd_scale
        print('\ninit area: ', self.area.eval(x0), ' --> opt area: ', self.area.eval(xs),
              '\ninit cl: ', cls0, ' --> opt cl: ', cls1,
              '\ninit cd: ', cds0, ' --> opt cd: ', cds1,
              '\ninit cl/cd: ', cls0 / cds0, ' --> opt cl/cd: ', cls1 / cds1,
              '\ninit ae: ', self.ae.eval(x0), ' --> opt ae: ', self.ae.eval(xs),
              '\ninit alfa: ', x0[-1], ' --> opt alfa: ', xs[-1]
              )
        # self.area.bs.show(xs)
        # input('wait ...')
        self.area.bs.show(x0, 2, MACH, '%s/init' % RESULT_DIR)
        self.area.bs.show(xs, 22, MACH, '%s/opt' % RESULT_DIR)


if __name__ == '__main__':
    def _init_fn(worker_id):
        np.random.seed(1 + worker_id)

    alfa = ALFA_MAX / 2.0

    data = np.load('data_bs/controls/%s.npy' % filename)
    data = np.hstack([data[1:33], data[35:-1][::-1], alfa])

    bs = BsplineArea(34)
    areaer = Areaer(bs=bs)

    net = ResNet1d18CLDMP(test=True)
    check = torch.load('mean-teacher-al/results-al-semi-%d/net_best_teacher.tar' % MACH)  # 150, 734
    net.load_state_dict(check['teacher'])
    net.cpu()
    net.eval()
    cfd = CFDNet(net=net)

    ae_net = AEcoder(ndfs=[4, 4, 8, 8], ngfs=[4, 4, 8, 8], embed_dim=16)
    check_ae = torch.load('shape-anomaly-detection/resultsAE_ID=16_wgan/ae_best_zeros.tar')
    ae_net.load_state_dict(check_ae['net_state_dict'])
    ae_net.cpu()
    ae_net.eval()
    ae = AENet(ae=ae_net, mu=check_ae['option']['mu'], sig=check_ae['option']['sig'])

    scales = check['scales']
    scale_cl = scales[0].item()
    scale_cd = scales[1].item()

    opter = ShapeOptConPts(cfd=cfd, ae=ae, area=areaer, cl_scale=scale_cl, cd_scale=scale_cd)
    alfa = torch.tensor(alfa).reshape(-1)
    opter.solve_minCD_conCL(data)

