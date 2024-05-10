import aerosandbox as asb
import aerosandbox.numpy as np
import os
import argparse
import time
import multiprocessing
from functools import partial
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=int, default=0, help='the id of job')
parser.add_argument('--start_ind', type=int, default=0, help='the start id of wing')
parser.add_argument('--end_ind', type=int, default=32, help='the end id of wing')
parser.add_argument('--num_paras', type=int, default=32, help='#paras per wing')
parser.add_argument('--key', type=int, default=0, help='0: train, 1: val, 2: test')


def cal_vol(airplane):
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


def generate_config_(paras):

    alfa, cr, ct, wl, span, theta, phi, psi, gamma, twist_r, twist_30, twist_70, twist_t, twist_w = paras

    theta, phi, psi, gamma = theta * np.pi / 180, phi * np.pi / 180, psi * np.pi / 180, gamma * np.pi / 180

    tan_theta = np.tan(theta)
    tan_phi = np.tan(phi)
    sin_gamma = np.sin(gamma)

    yt = span
    xt = tan_theta * yt + (cr - ct) / 4
    zt = yt * tan_phi

    ys30 = 0.3 * yt
    cs30 = (cr - (tan_theta * ys30 + cr / 4)) * 4 / 3
    xs30 = tan_theta * ys30 + (cr - cs30) / 4
    zs30 = ys30 * tan_phi

    cs70 = (3 * cs30 + 4 * ct) / 7.
    ys70 = 0.7 * yt
    xs70 = tan_theta * ys70 + (cr - cs70) / 4
    zs70 = ys70 * tan_phi

    cw = 0.5 * ct
    yw = yt + wl
    xw = xt + wl * sin_gamma
    zw = yw * tan_phi

    psi_ = np.pi / 2 - psi - phi
    cos_psi_ = np.cos(psi_)
    sin_psi_ = np.sin(psi_)
    rot = np.array([[cos_psi_, -sin_psi_],
                    [sin_psi_, cos_psi_]])
    ywt, zwt = rot @ np.array([yw - yt, zw - zt])
    yw = yt + ywt
    zw = zt + zwt

    return alfa, \
           ([0, 0, 0], cr, twist_r), \
           ([xs30, ys30, zs30], cs30, twist_30), \
           ([xs70, ys70, zs70], cs70, twist_70), \
           ([xt, yt, zt], ct, twist_t), \
           ([xw, yw, zw], cw, twist_w)


def generate_config_2(paras):

    alfa, cr, ct, wl, span, theta, phi, psi, gamma, twist_r, twist_30, twist_70, twist_t, twist_w = paras
    phi_, psi_1 = phi, psi
    theta, phi, psi, gamma = theta * np.pi / 180, phi * np.pi / 180, psi * np.pi / 180, gamma * np.pi / 180

    tan_theta = np.tan(theta)
    tan_phi = np.tan(phi)
    sin_gamma = np.sin(gamma)

    yt = span
    xt = tan_theta * yt + (cr - ct) / 4
    zt = yt * tan_phi

    ys30 = 0.3 * yt
    cs30 = (cr - (tan_theta * ys30 + cr / 4)) * 4 / 3
    xs30 = tan_theta * ys30 + (cr - cs30) / 4
    zs30 = ys30 * tan_phi

    cs70 = (3 * cs30 + 4 * ct) / 7.
    ys70 = 0.7 * yt
    xs70 = tan_theta * ys70 + (cr - cs70) / 4
    zs70 = ys70 * tan_phi

    cw = 0.5 * ct
    yw = yt + wl
    xw = xt + wl * sin_gamma
    zw = yw * tan_phi
    xw_ = xw
    yw_ = yw
    zw_ = zw

    psi_ = np.pi / 2 - psi - phi
    cos_psi_ = np.cos(psi_)
    sin_psi_ = np.sin(psi_)
    rot = np.array([[cos_psi_, -sin_psi_],
                    [sin_psi_, cos_psi_]])
    ywt, zwt = rot @ np.array([yw - yt, zw - zt])
    yw = yt + ywt
    zw = zt + zwt

    return alfa, \
           (0,    cr,   0,    phi_, twist_r), \
           (ys30, cs30, xs30, phi_, twist_30), \
           (ys70, cs70, xs70, phi_, twist_70), \
           (yt,   ct,   xt,   90 - psi_1, twist_t), \
           (yw_,  cw,   xw_, 90 - psi_1, twist_w)


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


def wing3D_analysis_mp(case_str,
                       velocity, altitude,
                       save_dir,
                       pts_data,
                       num_paras,
                       job_id):

    case = np.load(case_str, allow_pickle=True).item()
    a, b, c, d, e = case['wing']
    paras = case['paras']
    wing_name = 'job%d_%s_%d_%d_%d_%d_%d' % (job_id, case_str.split('/')[-1][:-4], a, b, c, d, e)

    airfoil_a = asb.Airfoil(coordinates=pts_data[a])
    airfoil_b = asb.Airfoil(coordinates=pts_data[b])
    airfoil_c = asb.Airfoil(coordinates=pts_data[c])
    airfoil_d = asb.Airfoil(coordinates=pts_data[d])
    airfoil_e = asb.Airfoil(coordinates=pts_data[e])
    res_np = np.zeros((num_paras, 17))
    for i in range(num_paras):

        alfa, para_a, para_b, para_c, para_d, para_e = generate_config_(paras[i])

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
                atmosphere=asb.Atmosphere(altitude),
                velocity=velocity,  # m/s
                alpha=alfa,  # degree
            )
        )

        aero = vlm.run()
        res_np[i, :14] = paras[i]
        res_np[i, 14] = aero['CL']
        res_np[i, 15] = aero['CD']
        res_np[i, 16] = cal_vol(airplane)

    np.save(save_dir + '/' + wing_name + '.npy', res_np)


if __name__ == '__main__':

    opt = parser.parse_args()
    opts = {k: v for k, v in opt._get_kwargs()}

    VELOCITY = 236.
    ALTITUDE = 12300.
    # conts_data = np.load('config/airfoil_conts_wgan.npy')
    pts_data = np.load('config/airfoil_pts_wgan.npy')
    KEYSET = ['train_set', 'val_set', 'test_set']

    BASE2 = 64
    KEY = KEYSET[opts['key']]
    DIR_WING_DATA = 'data_wing_rmmtw/' + KEY

    try:
        os.mkdir(DIR_WING_DATA)
    except OSError:
        pass

    try:
        case_list = ['config/paras_per_wing/%s' % KEY + '/' + it for it in os.listdir('config/paras_per_wing/%s' % KEY)]

    except FileNotFoundError:
        airfoils_perms = np.load('config/airfoils_permutations.npy', allow_pickle=True).item()
        paras_per_wing = np.load('config/paras_per_wing.npy')

        airfoils_perms_train = airfoils_perms['train_set']
        airfoils_perms_val = airfoils_perms['val_set']
        airfoils_perms_test = airfoils_perms['test_set']
        train_size = airfoils_perms_train.shape[0]
        val_size = airfoils_perms_val.shape[0]
        test_size = airfoils_perms_test.shape[0]

        paras_per_wing_train = paras_per_wing[:train_size*BASE2]
        paras_per_wing_val = paras_per_wing[train_size*BASE2:train_size*BASE2 + val_size*BASE2*2]
        paras_per_wing_test = paras_per_wing[train_size*BASE2 + val_size*BASE2*2:]
        try:
            os.mkdir('config/paras_per_wing/train_set')
            os.mkdir('config/paras_per_wing/val_set')
            os.mkdir('config/paras_per_wing/test_set')
        except OSError:
            pass

        cnt = 0
        for i in tqdm(range(train_size)):
            wing_i = airfoils_perms_train[i]
            paras_i = paras_per_wing_train[i*BASE2:(i+1)*BASE2]
            wing_paras_i = {'wing': wing_i, 'paras': paras_i}
            np.save('config/paras_per_wing/train_set/case%d.npy' % cnt, wing_paras_i)
            cnt += 1

        for i in tqdm(range(val_size)):
            wing_i = airfoils_perms_val[i]
            paras_i = paras_per_wing_val[i*BASE2*2:(i+1)*BASE2*2]
            wing_paras_i = {'wing': wing_i, 'paras': paras_i}
            np.save('config/paras_per_wing/val_set/case%d.npy' % cnt, wing_paras_i)
            cnt += 1

        for i in tqdm(range(test_size)):
            wing_i = airfoils_perms_test[i]
            paras_i = paras_per_wing_test[i*BASE2*2:(i+1)*BASE2*2]
            wing_paras_i = {'wing': wing_i, 'paras': paras_i}
            np.save('config/paras_per_wing/test_set/case%d.npy' % cnt, wing_paras_i)
            cnt += 1

        case_list = ['config/paras_per_wing/%s' % KEY + '/' + it for it in os.listdir('config/paras_per_wing/%s' % KEY)]

    print('=========================start generating=====================')
    t0 = time.perf_counter()
    wing3D_analysis_mp_ = partial(wing3D_analysis_mp,
                                  velocity=VELOCITY, altitude=ALTITUDE,
                                  save_dir=DIR_WING_DATA,
                                  pts_data=pts_data,
                                  num_paras=opts['num_paras'],
                                  job_id=opts['job_id'])

    case_list = sorted(case_list, key=lambda it: int(it.split('/')[-1][:-4][4:]))
    case_list = case_list[opts['start_ind']:opts['end_ind']]
    # for it in tqdm(case_list):
    #     wing3D_analysis_mp_(it)
    with multiprocessing.Pool(3) as pool:
        r = list(tqdm(pool.imap(wing3D_analysis_mp_, case_list), total=len(case_list)))
    t1 = time.perf_counter()
    print('=========================end generating=====================')
    print('cost: ', t1 - t0)

