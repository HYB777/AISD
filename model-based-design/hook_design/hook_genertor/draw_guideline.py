from bspline_utils import *
import mpl_toolkits.axisartist as axisartist


def draw_guideline_(params):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    degree = 3
    T = 50

    SQRT2 = 2 ** 0.5
    R = 0.025
    THETAS = np.linspace(-3, -1, 100) / 4 * np.pi
    ARC = [R * np.cos(THETAS), R * np.sin(THETAS) + R]
    plt.plot(ARC[0], ARC[1], 'grey', label='quarter circle (where R=0.025)')
    LPTS = [-R / SQRT2, (1 - 1 / SQRT2) * R]
    RPTS = [R / SQRT2, (1 - 1 / SQRT2) * R]
    plt.text(0 - 0.006, R + 0.0025, '(0, R)', fontsize=13)

    n_controls1 = 3
    n_controls2 = 7
    knots1 = get_knots(degree=degree - 1, n_controls=n_controls1)
    knots2 = get_knots(degree=degree, n_controls=n_controls2)
    guide_pts = params['guide_pts'][0]

    ptsX1 = np.array([guide_pts[0], guide_pts[0], LPTS[0]])
    ptsY1 = np.array([guide_pts[1], -guide_pts[0] + (1 - SQRT2) * R, LPTS[1]])

    plt.text(guide_pts[0] - 0.005, guide_pts[1] + 0.005, 'A ($g_1, g_2$)', fontsize=13)
    plt.text(guide_pts[0] - 0.05, -guide_pts[0] + (1 - SQRT2) * R, '($g_1, -g_1+(1-\sqrt{2}R)$)', fontsize=13)
    plt.text(LPTS[0] - 0.04, LPTS[1] - 0.005, 'B ($-R/\sqrt{2}, (1-1/\sqrt{2})R$)', fontsize=13)
    plt.text(0 - 0.006, 0 - 0.005, '(0, 0)', fontsize=13)

    try:
        N1 = np.load('N1_c%d_t%d.npy' % (n_controls1, T), allow_pickle=True).item()
    except FileNotFoundError:
        N1 = get_N(n_controls=n_controls1, knots=knots1, T=T, degree=degree - 1)
        np.save('N1_c%d_t%d.npy' % (n_controls1, T), N1)

    scales = params['scales'][0]
    g8 = -0.5 * scales[-1]
    x_hat1 = N1 @ ptsX1
    y_hat1 = N1 @ ptsY1

    ptsX2 = np.array([RPTS[0], guide_pts[2], guide_pts[3], guide_pts[5], guide_pts[6], g8, g8])
    ptsY2 = np.array([RPTS[1], guide_pts[2] + (1 - SQRT2) * R, guide_pts[4], 0, 0, 0, guide_pts[-1]])
    ptsY2[3:-1] = np.linspace(guide_pts[4], guide_pts[-1], 5)[1:-1]

    plt.text(RPTS[0] + 0.000, RPTS[1] - 0.005, 'C ($R/\sqrt{2}, (1-1/\sqrt{2})R$)', fontsize=13)
    plt.text(guide_pts[2] + 0.000, guide_pts[2] + (1 - SQRT2) * R - 0.005,
             '($g_3, g_3+(1-\sqrt{2}R)$)', fontsize=13)
    plt.text(guide_pts[3] + 0.001, guide_pts[4], '($g_4, g_5$)', fontsize=13)
    plt.text(guide_pts[5] + 0.001, ptsY2[3], '($g_6, 0.25g_8+0.75g_5$)', fontsize=13)
    plt.text(guide_pts[6] + 0.001, ptsY2[4], '($g_7, 0.5g_8+0.5g_5$)', fontsize=13)
    plt.text(g8 + 0.001, ptsY2[5], r'($-\frac{s_{11}}{2}, 0.75g_8+0.25g5$)', fontsize=13)
    plt.text(g8 - 0.009, guide_pts[-1] + 0.003, r'E ($-\frac{s_{11}}{2}, g_8$)', fontsize=13)

    try:
        N2 = np.load('N2_c%d_t%d.npy' % (n_controls2, T), allow_pickle=True).item()
    except FileNotFoundError:
        N2 = get_N(n_controls=n_controls2, knots=knots2, T=T, degree=degree)
        np.save('N2_c%d_t%d.npy' % (n_controls2, T), N2)
    x_hat2 = N2 @ ptsX2
    y_hat2 = N2 @ ptsY2

    plt.plot(ptsX1, ptsY1, 'go-', label='control points')
    plt.plot(x_hat1, y_hat1, 'r-', label='guideline')
    plt.plot(ptsX2, ptsY2, 'go-')
    plt.plot(x_hat2, y_hat2, 'r-')
    t = np.linspace(0, 1, 100)

    plt.plot(t * -1 / SQRT2 * R, R + t * -1 / SQRT2 * R, c='grey', linestyle='--')
    plt.plot(t * 1 / SQRT2 * R, R + t * -1 / SQRT2 * R, c='grey', linestyle='--')
    plt.plot([-1 / SQRT2 * R * 0.2, 0], [R + 0.2 * -1 / SQRT2 * R, (1 - 0.2 * SQRT2) * R], 'grey')
    plt.plot([1 / SQRT2 * R * 0.2, 0], [R + 0.2 * -1 / SQRT2 * R, (1 - 0.2 * SQRT2) * R], 'grey')

    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["x"].set_axisline_style("-|>", size=1)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("-|>", size=1)
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")
    plt.axis('equal')
    plt.show()


xs = scio.loadmat('../optimized_results/opt_minSE_conV_w0.01/x_opt.mat')
draw_guideline_(xs)