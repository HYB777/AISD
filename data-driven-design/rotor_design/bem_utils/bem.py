import numpy as np
from math import radians, degrees, sqrt, cos, sin, atan2, atan, pi, acos, exp
import pandas as pd
from scipy import optimize
from pybemt.airfoil import load_airfoil, Airfoil
from scipy.interpolate import interp1d



"""
cfg = {
    'case': {
        'rpm':
        'v_inf':
    }
    '
}
"""

# AR = 7.32/0.53
N_ALPHA = 181


def viterna_(alpha, cl_adj, cdmax, A, B):

    alpha = np.maximum(alpha, 0.0001)  # prevent divide by zero

    cl = cdmax / 2 * np.sin(2 * alpha) + A * np.cos(alpha) ** 2 / np.sin(alpha)
    cl = cl * cl_adj

    cd = cdmax * np.sin(alpha) ** 2 + B * np.cos(alpha)

    return cl, cd


def table_extrapolation(cald, AR):
    cald_min = cald[0]
    cald_max = cald[-1]
    alpha_high = radians(cald_max[0])
    cl_high = cald_max[1]
    cd_high = cald_max[2]
    alpha_low = radians(cald_min[0])
    cl_low = cald_min[1]
    cd_low = cald_min[2]
    if alpha_high > pi / 2:
        raise Exception('alpha_high > pi / 2 or alpha_high <= 0')
    if alpha_low < -pi / 2:
        raise Exception('alpha_low < -pi / 2 or alpha_low >= 0')
    # if alpha_high > pi / 2 or alpha_high <= 0:
    #     raise Exception('alpha_high > pi / 2 or alpha_high <= 0')
    # if alpha_low < -pi / 2 or alpha_low >= 0:
    #     raise Exception('alpha_low < -pi / 2 or alpha_low >= 0')# ct=0.0262061572, cq=0.0011810374, eta=0.6127064626
    sa = sin(alpha_high)
    ca = cos(alpha_high)
    cdmax = 1.11 + 0.018 * AR
    cl_adj = 0.7
    A = (cl_high - cdmax * sa * ca) * sa / ca ** 2
    B = (cd_high - cdmax * sa * sa)
    n_alpha = int((90 - cald_max[0])/0.5)+1
    alpha1 = np.linspace(alpha_high, pi / 2, n_alpha)
    alpha1 = alpha1[1:]  # remove first element so as not to duplicate when concatenating
    cl1, cd1 = viterna_(alpha1, 1.0, cdmax, A, B)

    alpha2 = np.linspace(pi / 2, pi - alpha_high, n_alpha)
    alpha2 = alpha2[1:]
    cl2, cd2 = viterna_(pi - alpha2, -cl_adj, cdmax, A, B)

    alpha3 = np.linspace(pi - alpha_high, pi, N_ALPHA - n_alpha + 1)
    alpha3 = alpha3[1:]
    cl3, cd3 = viterna_(pi - alpha3, 1.0, cdmax, A, B)
    cl3 = (alpha3 - pi) / alpha_high * cl_high * cl_adj  # override with linear variation

    if alpha_low <= -alpha_high:
        alpha4 = []
        cl4 = []
        cd4 = []
        alpha5max = alpha_low
        deg5max = cald_min[0]
    else:
        # -alpha_high <-> alpha_low
        # Note: this is done slightly differently than AirfoilPrep for better continuity

        alpha4 = np.linspace(-alpha_high, alpha_low, int((cald_min[0] + cald_max[0]) / 0.5) + 1)
        alpha4 = alpha4[1:-2]  # also remove last element for concatenation for this case
        cl4 = -cl_high * cl_adj + (alpha4 + alpha_high) / (alpha_low + alpha_high) * (cl_low + cl_high * cl_adj)
        cd4 = cd_low + (alpha4 - alpha_low) / (-alpha_high - alpha_low) * (cd_high - cd_low)
        alpha5max = -alpha_high
        deg5max = -cald_max[0]

    # -90 <-> -alpha_high
    alpha5 = np.linspace(-pi / 2, alpha5max, int((deg5max + 90) / 0.5) + 1)
    alpha5 = alpha5[1:]
    cl5, cd5 = viterna_(-alpha5, -cl_adj, cdmax, A, B)

    # -180+alpha_high <-> -90
    alpha6 = np.linspace(-pi + alpha_high, -pi / 2, int((-90 - (-180 + cald_max[0])) / 0.5) + 1)
    alpha6 = alpha6[1:]
    cl6, cd6 = viterna_(alpha6 + pi, cl_adj, cdmax, A, B)

    # -180 <-> -180 + alpha_high
    alpha7 = np.linspace(-pi, -pi + alpha_high, int(((-180 + cald_max[0]) + 180) / 0.5) + 1)
    cl7, cd7 = viterna_(alpha7 + pi, 1.0, cdmax, A, B)
    cl7 = (alpha7 + pi) / alpha_high * cl_high * cl_adj  # linear variation

    cdmin = 0.001
    alpha = np.concatenate((alpha7, alpha6, alpha5, alpha4, np.radians(cald[1:, 0]), alpha1, alpha2, alpha3))
    deg = np.rad2deg(alpha)
    cl = np.concatenate((cl7, cl6, cl5, cl4, cald[1:, 1], cl1, cl2, cl3))
    cd = np.concatenate((cd7, cd6, cd5, cd4, cald[1:, 2], cd1, cd2, cd3))

    cd = np.maximum(cd, cdmin)  # don't allow negative drag coefficients

    acld = np.hstack([deg.reshape(-1, 1), cl.reshape(-1, 1), cd.reshape(-1, 1)])
    # print(np.sum(np.abs(deg[:-1]-deg[1:]) == 0))
    # plt.plot(deg, cl, 'r')
    # plt.plot(deg, cd, 'b')
    # plt.show()
    # print(np.sum((acld[1:, 0] - acld[:-1, 0]) == 0))

    return acld


def my_load_airfoil(s, i=-1, AR=-1):
    if s.startswith('case'):
        a = Airfoil()
        cald = np.load('../airfoils/cases0123/%s.npy' % s, allow_pickle=True).item()[i]
        table_ = table_extrapolation(cald, AR)
        a.name = s
        a.alpha_, a.Cl_, a.Cd_ = table_[:, 0], table_[:, 1], table_[:, 2]
        a.Cl_func = interp1d(a.alpha_, a.Cl_, kind='quadratic')
        a.Cd_func = interp1d(a.alpha_, a.Cd_, kind='quadratic')
        # acld = np.hstack([a.alpha_[:, np.newaxis], a.Cl_[:, np.newaxis], a.Cd_[:, np.newaxis],  a.Cd_[:, np.newaxis]])
        # np.savetxt('%s_acld.dat'%s, acld)
    else:
        a = load_airfoil(s)
    return a


def my_load_airfoil_(cald, name, AR=15.230080238850533):
    a = Airfoil()
    table_ = table_extrapolation(cald, AR)
    a.name = name
    a.alpha_, a.Cl_, a.Cd_ = table_[:, 0], table_[:, 1], table_[:, 2]
    a.Cl_func = interp1d(a.alpha_, a.Cl_, kind='quadratic')
    a.Cd_func = interp1d(a.alpha_, a.Cd_, kind='quadratic')
    return a


class MyFluid:

    def __init__(self, cfg):
        self.rho = cfg['rho']
        self.mu = cfg['mu']


class MySection:

    def __init__(self, airfoil, radius, width, pitch, chord, rotor, mode):
        self.airfoil = airfoil
        self.radius = radius
        self.width = width
        self.pitch = pitch
        self.chord = chord
        self.rotor = rotor

        if mode == 'turbine':
            self.C = -1
        else:
            self.C = 1

        self.v = 0.0
        self.v_theta = 0.0
        self.v_rel = 0.0
        self.a = 0.0
        self.ap = 0.0
        self.Re = 0.0
        self.alpha = 0.0
        self.AoA = 0.0
        self.dT = 0.0
        self.dQ = 0.0
        self.F = 0.0
        self.Cl = 0.0
        self.Cd = 0.0

        self.precalc()

    def precalc(self):
        """
        Calculation of properties before each solver run, to ensure all parameters are correct for parameter sweeps.

        :return: None
        """
        self.sigma = self.rotor.n_blades * self.chord / (2 * pi * self.radius)

    def tip_loss(self, phi):
        """
        Prandtl tip loss factor, defined as

        .. math::
            F = \\frac{2}{\\pi}\\cos^{-1}e^{-f} \\\\
            f = \\frac{B}{2}\\frac{R-r}{r\\sin\\phi}

        A hub loss is also caluclated in the same manner.

        :param float phi: Inflow angle
        :return: Combined tip and hub loss factor
        :rtype: float
        """

        def prandtl(dr, r, phi):
            f = self.rotor.n_blades * dr / (2 * r * (sin(phi)))
            if (-f > 500):  # exp can overflow for very large numbers
                F = 1.0
            else:
                F = 2 * acos(min(1.0, exp(-f))) / pi

            return F

        if phi == 0:
            F = 1.0
        else:
            r = self.radius
            Ftip = prandtl(self.rotor.blade_radius - r, r, phi)
            Fhub = prandtl(r - self.rotor.radius_hub, r, phi)
            F = Ftip * Fhub

        self.F = F
        return F

    def stall_delay_model(self, phi, alpha, Cl, Cd):
        """
        The 3D correction model based on Chaviaropoulos and Hansen ref:

        .. bib::
            @article{chaviaropoulos2000investigating,
                     title={Investigating three-dimensional and rotational effects on wind turbine blades by means of a quasi-3D Navier-Stokes solver},
                     author={Chaviaropoulos, PK and Hansen, Martin OL},
                     journal={J. Fluids Eng.},
                     volume={122},
                     number={2},
                     pages={330--336},
                     year={2000}
                     }

        .. math::
            Cl_3D = Cl_2D + a (c / r)^h \cos^n{twist} (Cl_inv - Cl_2d) \\
            Cl_inv = \sqrt{Cl_2d^2 + Cd^2}
        where:
        a = 2.2, h = 1.3 and n = 4

        :param float phi: Inflow angle
        :return: Lift coefficient with 3D correction
        :rtype: float
        """
        Cl_inv = sqrt(Cl ** 2 + Cd ** 2)
        twist = alpha - self.pitch
        r = self.radius - self.rotor.radius_hub
        c = self.chord

        a = 2.2
        h = 1.3
        n = 4
        return Cl + a * (c / r) ** h * (cos(twist)) ** n * (Cl_inv - Cl)

    def airfoil_forces(self, phi):
        """
        Force coefficients on an airfoil, decomposed in axial and tangential directions:

        .. math::
            C_T = C_l\\cos{\\phi} - CC_d\\sin{\\phi} \\\\
            C_Q = C_l\\sin{\\phi} + CC_d\\cos{\\phi} \\\\

        where drag and lift coefficients come from
        airfoil tables.

        :param float phi: Inflow angle
        :return: Axial and tangential force coefficients
        :rtype: tuple
        """

        C = self.C

        alpha = C * (self.pitch - phi)

        Cl = self.airfoil.Cl(alpha)
        Cd = self.airfoil.Cd(alpha)

        CT = Cl * cos(phi) - C * Cd * sin(phi)
        CQ = Cl * sin(phi) + C * Cd * cos(phi)

        self.AoA = degrees(alpha)
        self.Cl = float(Cl)
        self.Cd = float(Cd)

        return CT, CQ

    def induction_factors(self, phi):
        """
        Calculation of axial and tangential induction factors,

        .. math::
            a = \\frac{1}{\\kappa - C} \\\\
            a\' = \\frac{1}{\\kappa\' + C} \\\\
            \\kappa = \\frac{4F\\sin^2{\\phi}}{\\sigma C_T} \\\\
            \\kappa\' = \\frac{4F\\sin{\\phi}\\cos{\\phi}}{\\sigma C_Q} \\\\

        :param float phi: Inflow angle
        :return: Axial and tangential induction factors
        :rtype: tuple
        """

        C = self.C

        F = self.tip_loss(phi)

        CT, CQ = self.airfoil_forces(phi)

        kappa = 4 * F * sin(phi) ** 2 / (self.sigma * CT)
        kappap = 4 * F * sin(phi) * cos(phi) / (self.sigma * CQ)

        a = 1.0 / (kappa - C)
        ap = 1.0 / (kappap + C)

        return a, ap

    def func(self, phi, v_inf, omega):
        """
        Residual function used in root-finding functions to find the inflow angle for the current section.

        .. math::
            \\frac{\\sin\\phi}{1+Ca} - \\frac{V_\\infty\\cos\\phi}{\\Omega R (1 - Ca\')} = 0\\\\

        :param float phi: Estimated inflow angle
        :param float v_inf: Axial inflow velocity
        :param float omega: Tangential rotational velocity
        :return: Residual
        :rtype: float
        """
        # Function to solve for a single blade element
        C = self.C

        a, ap = self.induction_factors(phi)
        # print(1 + C * a, omega * self.radius * (1 - C * ap))
        resid = sin(phi) / (1 + C * a) - v_inf * cos(phi) / (omega * self.radius * (1 - C * ap))

        self.a = a
        self.ap = ap

        return resid

    def forces(self, phi, v_inf, omega, fluid):
        """
        Calculation of axial and tangential forces (thrust and torque) on airfoil section.

        The definition of blade element theory is used,

        .. math::
            \\Delta T = \\sigma\\pi\\rho U^2C_T r\\Delta r \\\\
            \\Delta Q = \\sigma\\pi\\rho U^2C_Q r^2\\Delta r \\\\
            U = \\sqrt{v^2+v\'^2} \\\\
            v = (1 + Ca)V_\\infty \\\\
            v\' = (1 - Ca\')\\Omega R \\\\

        Note that this is equivalent to the momentum theory definition,

        .. math::
            \\Delta T = 4\\pi\\rho r V_\\infty^2(1 + Ca)aF\\Delta r \\\\
            \\Delta Q = 4\\pi\\rho r^3 V_\\infty\\Omega(1 + Ca)a\'F\\Delta r \\\\


        :param float phi: Inflow angle
        :param float v_inf: Axial inflow velocity
        :param float omega: Tangential rotational velocity
        :param Fluid fluid: Fluid
        :return: Axial and tangential forces
        :rtype: tuple
        """

        C = self.C
        r = self.radius
        rho = fluid.rho

        a, ap = self.induction_factors(phi)
        CT, CQ = self.airfoil_forces(phi)

        v = (1 + C * a) * v_inf
        vp = (1 - C * ap) * omega * r
        U = sqrt(v ** 2 + vp ** 2)

        self.Re = rho * U * self.chord / fluid.mu

        # From blade element theory
        self.dT = self.sigma * pi * rho * U ** 2 * CT * r * self.width
        self.dQ = self.sigma * pi * rho * U ** 2 * CQ * r ** 2 * self.width

        # From momentum theory
        # dT = 4*pi*rho*r*self.v_inf**2*(1 + a)*a*F
        # dQ = 4*pi*rho*r**3*self.v_inf*(1 + a)*ap*self.omega*F

        return self.dT, self.dQ


class MyRotor:

    def __init__(self, airfoils, mode, cfg):
        self.n_blades = cfg['nblades']
        self.diameter = cfg['diameter']

        s = airfoils
        c = cfg['chord']
        r = cfg['radius']
        dr = cfg['dr']
        self.n_sections = len(s)
        AR = (self.diameter/2)**2/(np.sum(np.array(dr)*np.array(c))+cfg['radius_hub']*c[0])
        AR = 15.230080238850533

        self.alpha = cfg['pitch']
        self.sections = []
        for i in range(self.n_sections):
            sec = MySection(my_load_airfoil(s[i], i, AR), r[i], dr[i], radians(self.alpha[i]), c[i], self, mode)
            self.sections.append(sec)

        self.radius_hub = cfg['radius_hub']

        self.precalc(twist=0.0)

    def precalc(self, twist):
        """
        Calculation of properties before each solver run, to ensure all parameters are correct for parameter sweeps.

        :return: None
        """
        self.blade_radius = 0.5 * self.diameter
        self.area = pi * self.blade_radius ** 2

        # Apply twist
        for i, sec in enumerate(self.sections):
            sec.pitch = radians(self.alpha[i] + twist)

    def sections_dataframe(self):
        """
        Creates a pandas DataFrame with all calculated section properties.

        :return: DataFrame with section properties
        :rtype: pd.DataFrame
        """

        columns = ['radius', 'chord', 'pitch', 'Cl', 'Cd', 'dT', 'dQ', 'F', 'a', 'ap', 'Re', 'AoA']
        data = {}
        for param in columns:
            array = [getattr(sec, param) for sec in self.sections]
            data[param] = array

        return pd.DataFrame(data)


class MyRotorSolver:

    def __init__(self, airfoils, cfg):
        self.name = '_'.join(airfoils) + '_' + str(cfg['pitch'])
        # Case
        self.v_inf = cfg['v_inf']
        self.rpm = cfg['rpm']
        self.twist = cfg['twist']
        self.coaxial = cfg['coaxial']
        self.mode = cfg['mode']  # turbine or rotor, default rotor
        self.rotor = MyRotor(airfoils, self.mode, cfg)

        # Fluid
        self.fluid = MyFluid(cfg)

        # Output
        self.T = 0  # Thrust
        self.Q = 0  # Torque
        self.P = 0  # Power

        # Coaxial
        assert not self.coaxial
        if self.coaxial:
            self.rpm2 = cfg['rpm2']
            self.twist2 = cfg['twist2']
            self.rotor2 = MyRotor(airfoils, self.mode, cfg)
            self.zD = cfg['dz'] / self.rotor.diameter
            self.T2 = 0
            self.Q2 = 0
            self.P2 = 0

        # Solver
        self.solver = 'bisect'
        self.Cs = 0.625
        # if cfg.has_section('solver'):
        #     self.solver = cfg.get('solver', 'solver')
        #     if cfg.has_option('solver', 'Cs'):
        #         self.Cs = cfg.getfloat('solver', 'Cs')

    def rotor_coeffs(self, T, Q, P):
        """
        Dimensionless coefficients for a rotor.

        .. math::
            \\text{J} = \\frac{V_\\infty}{nD} \\\\
            C_T = \\frac{T}{\\rho n^2 D^4} \\\\
            C_Q = \\frac{Q}{\\rho n^2 D^5} \\\\
            C_P = 2\\pi C_Q \\\\
            \\eta = \\frac{C_T}{C_P}J \\\\

        :param float T: Thrust
        :param float Q: Torque
        :param float P: Power
        :return: Advance ratio, thrust coefficient, torque coefficient, power coefficient and efficiency
        :rtype: tuple
        """

        D = self.rotor.diameter
        R = 0.5 * D
        rho = self.fluid.rho
        n = self.rpm / 60.0
        J = self.v_inf / (n * D)
        omega = self.rpm * 2 * pi / 60.0

        CT = T / (rho * n ** 2 * D ** 4)
        CQ = Q / (rho * n ** 2 * D ** 5)
        CP = 2 * pi * CQ

        if J == 0.0:
            eta = (CT / CP)
        else:
            eta = (CT / CP) * J

        return J, CT, CQ, CP, eta

    def turbine_coeffs(self, T, Q, P):
        """
        Dimensionless coefficients for a turbine.

        .. math::
            \\text{TSR} = \\frac{\\Omega R}{V_\\infty} \\\\
            C_T = \\frac{2T}{\\rho A V_\\infty^2} \\\\
            C_P = \\frac{2P}{\\rho A V_\\infty^3} \\\\

        :param float T: Thrust
        :param float Q: Torque
        :param float P: Power
        :return: Tip-speed ratio, power coefficient and thrust coefficient
        :rtype: tuple
        """

        rho = self.fluid.rho
        V = self.v_inf
        omega = self.rpm * 2 * pi / 60.0
        TSR = omega * self.rotor.blade_radius / V
        CT = T / (0.5 * rho * self.rotor.area * V ** 2)
        CP = P / (0.5 * rho * self.rotor.area * V ** 3)

        return TSR, CP, CT

    def run_sweep(self, parameter, n, low, high):
        """
        Utility function to run a sweep of a single parameter.

        :param string parameter: Parameter to sweep, must be a member of the Solver class.
        :param int n: Number of runs
        :param float low: Minimum parameter value
        :param float high: Maximum parameter value

        :return: DataFrame of results and list of sections for each run
        :rtype: tuple
        """

        if self.mode == 'turbine':
            df = pd.DataFrame(columns=[parameter, 'T', 'Q', 'P', 'TSR', 'CT', 'CP'], index=range(n))
        else:
            if self.coaxial:
                cols = [parameter, 'T', 'Q', 'P', 'J', 'CT', 'CQ', 'CP', 'eta',
                        'CT2', 'CQ2', 'CP2', 'eta2']
            else:
                cols = [parameter, 'T', 'Q', 'P', 'J', 'CT', 'CQ', 'CP', 'eta']

            df = pd.DataFrame(columns=cols, index=range(n))

        sections = []
        for i, p in enumerate((np.linspace(low, high, n))):
            setattr(self, parameter, p)

            if self.mode == 'turbine':
                T, Q, P, sec_df = self.run()
                TSR, CP, CT = self.turbine_coeffs(T, Q, P)
                df.iloc[i] = [p, T, Q, P, TSR, CT, CP]
            else:
                if self.coaxial:
                    T, Q, P, sec_df, T2, Q2, P2, sec_df2 = self.run()
                    J, CT, CQ, CP, eta = self.rotor_coeffs(T, Q, P)
                    J, CT2, CQ2, CP2, eta = self.rotor_coeffs(T2, Q2, P2)
                    df.iloc[i] = [p, T, Q, P, T2, Q2, P2, J, CT, CQ, CP, eta, CT2, CP2, eta2]
                else:
                    T, Q, P, sec_df = self.run()
                    J, CT, CQ, CP, eta = self.rotor_coeffs(T, Q, P)
                    df.iloc[i] = [p, T, Q, P, J, CT, CQ, CP, eta]

            sections.append(sec_df)

        return df, sections

    def solve(self, rotor, twist, rpm, v_inflow, r_inflow):
        """
        Find inflow angle and calculate forces for a single rotor given rotational speed, inflow velocity and radius.

        :param Rotor rotor: Rotor to solve for
        :param float twist: Angle to adjust rotor pitch
        :param float rpm: Rotations per minute
        :param float v_inflow: Inflow velocity
        :param float r_inflow: Inflow radius (equal to blade radius for single rotors)
        :return: Calculated thrust, torque and power for the rotor
        :rtype: tuple
        """

        rotor.precalc(twist)

        omega = rpm * 2 * pi / 60.0
        # Axial momentum (thrust)
        T = 0.0
        # Angular momentum
        Q = 0.0
        flag = True
        for sec in rotor.sections:
            if sec.radius < r_inflow:
                v = v_inflow
            else:
                v = 0.0

            if self.solver == 'brute':
                phi = self.brute_solve(sec, v, omega)
            else:
                try:
                    phi = optimize.bisect(sec.func, 0.01 * pi, 0.9 * pi, args=(v, omega))
                except ValueError as e:
                    print(self.name)
                    print(e)
                    print('Bisect failed, switching to brute solver')
                    phi = self.brute_solve(sec, v, omega)

            dT, dQ = sec.forces(phi, v, omega, self.fluid)

            # Integrate
            T += dT
            Q += dQ

        # Power
        P = Q * omega

        return T, Q, P

    def slipstream(self):
        """
        For coaxial calculations. Calculates slipstream radius and velocity for the upper rotor according to
        momentum theory. Currently only the static case is included.

        .. math::
            r_s = \\frac{R}{\\sqrt{2}} \\\\
            v_s = C_s\\sqrt{\\frac{2 T}{\\rho A}} \\\\

        :return: Radius and velocity of the slipstream
        :rtype: tuple
        """

        r_s = self.rotor.blade_radius / sqrt(2.0)
        v_s = self.Cs * sqrt(2 * self.T / (self.fluid.rho * self.rotor.area))

        return r_s, v_s

    def run(self):
        """
        Runs the solver, i.e. finds the forces for each rotor.

        :return: Calculated thrust, torque, power and DataFrame with properties for all sections.
        :rtype: tuple
        """
        self.T, self.Q, self.P = self.solve(self.rotor, self.twist, self.rpm, self.v_inf, self.rotor.diameter)
        # print('--- Results ---')
        # print('Trust (N):\t', self.T)
        # print('Torque (Nm):\t', self.Q)
        # print('Power (W):\t', self.P)

        # Coaxial calculaction
        if self.coaxial:
            self.r_s, self.v_s = self.slipstream()

            self.T2, self.Q2, self.P2 = self.solve(self.rotor2, self.twist2, self.rpm2, self.v_s, self.r_s)

            # print('Trust 2 (N):\t', self.T2)
            # print('Torque 2 (Nm):\t', self.Q2)
            # print('Power 2 (W):\t', self.P2)

            return self.T, self.Q, self.P, self.rotor.sections_dataframe(), self.T2, self.Q2, self.P2, self.rotor2.sections_dataframe()

        else:
            return self.T, self.Q, self.P, self.rotor.sections_dataframe()

    def brute_solve(self, sec, v, omega, n=3600):
        """
        Solve by a simple brute force procedure, iterating through all
        possible angles and selecting the one with lowest residual.

        :param Section sec: Section to solve for
        :param float v: Axial inflow velocity
        :param float omega: Tangential rotational velocity
        :param int n: Number of angles to test for, optional
        :return: Inflow angle with lowest residual
        :rtype: float
        """
        resid = np.zeros(n)
        phis = np.linspace(-0.9 * np.pi, 0.9 * np.pi, n)
        for i, phi in enumerate(phis):
            res = sec.func(phi, v, omega)
            if not np.isnan(res):
                resid[i] = res
            else:
                resid[i] = 1e30
        i = np.argmin(abs(resid))
        return phis[i]

    def optimize_pitch(self):
        """
        Optimize rotor pitch for either maximum thrust (propeller) or maximum power (turbine)
        using a genetic evolution algorithm.

        This is intended as an example of how optimization can be done using the scipy.optimize
        package. The overall procedure can be readily modified to optimize for other parameters,
        e.g. a parametrized function for the pitch, or for a parameter sweep instead of
        a single parameter set.

        return: Array of optimized pitches
        """

        def run_bemt(x):
            print('Current iteration:', x)
            for sec, pitch in zip(self.rotor.sections, x):
                sec.pitch = np.radians(pitch)

            T, Q, P, df = self.run()
            J, CT, CQ, CP, eta = s.rotor_coeffs(T, Q, P)
            if self.mode == 'turbine':
                return -P
            else:
                return -eta

        x = [sec.pitch for sec in self.rotor.sections]
        bounds = [(0, 30)] * len(x)

        result = optimize.differential_evolution(run_bemt, bounds, tol=1e-1)

        return result


if __name__ == '__main__':
    """
    [case]
    rpm = 1100.0
    v_inf = 1.0
    [rotor]
    nblades = 3
    diameter = 3.054
    radius_hub = 0.375
    section = CLARKY CLARKY CLARKY CLARKY CLARKY CLARKY CLARKY 
    radius = 0.525 0.675 0.825 0.975 1.125 1.275 1.425
    chord  = 0.18 0.225 0.225 0.21 0.1875 0.1425 0.12
    pitch = 17 17 17 17 17 17 17 
    dr = 0.15 0.15 0.15 0.15 0.15 0.15 0.15
    [fluid]
    rho=1.225
    mu=1.81e-5
    """
    from math import pi

    cfg = {
        'rpm': 300,
        'v_inf': 12.7,
        'twist': 10,
        'mode': 'rotor',
        'nblades': 2,
        'diameter': 7.32*2,
        'radius_hub': 0.6,
        'radius': [0.61, 1.70, 4.77, 7.00],
        'chord': [0.57, 0.60, 0.50, 0.44],
        'pitch': [11.8,  1.016,  0.859, -0.007],  # [5.536e+00,  8.117e+00, 1.078e+01, 2.404e+01],
        # 'pitch': [20.078125, 18.984375, 4.4375, -1.375],
        'dr': [0.545, 2.08, 2.65, 1.115+0.32],
        # 'dr': [1.7-0.61, 4.77-1.7, 7-4.77, 0.32],
        'rho': 1.19252,
        'mu': 1.81e-5,
        'coaxial': False,
    }
    airfoils_list = ['case2822', 'case17645', 'case24913', 'case22967']
    # airfoils_list = ['case29110', 'case11783', 'case28886', 'case28522']
    s = MyRotorSolver(airfoils_list, cfg)
    T, Q, P, _ = s.run()
    J, CT, CQ, CP, eta = s.rotor_coeffs(T, Q, P)
    # J, CT, CQ, CP, eta = s.turbine_coeffs(T, Q, P)
    # print(T, Q, P)
    # TSR, CP, CT = s.turbine_coeffs(T, Q, P)
    # print(s.turbine_coeffs(T, Q, P), (CT/CP)*3.14/TSR)
    print(T, Q, P, eta, CT, CQ, CP, J)
    print(eta, CT, CQ, CP, J)
    # print(s.optimize_pitch())