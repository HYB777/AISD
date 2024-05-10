from .bem import *
from .pyxfoil import *
from .bs_utils import BsplineArea
import pybemt

class MyRotorBeta:

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
            sec = MySection(my_load_airfoil_(s[i], i, AR), r[i], dr[i], radians(self.alpha[i]), c[i], self, mode)
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


class MyRotorSolverBeta:

    def __init__(self, airfoils, cfg):
        self.name = 'custom'
        # Case
        self.v_inf = cfg['v_inf']
        self.rpm = cfg['rpm']
        self.twist = cfg['twist']
        self.coaxial = cfg['coaxial']
        self.mode = cfg['mode']  # turbine or rotor, default rotor
        self.rotor = MyRotorBeta(airfoils, self.mode, cfg)

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
            self.rotor2 = MyRotorBeta(airfoils, self.mode, cfg)
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
                    raise NotImplementedError
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


def rotor_cal(sections, n_blade, pitch):
    foil = [xfoil_cal(sections[i], i) for i in range(4)]

    cfg = {
        'rpm': 300,
        'v_inf': 12.7,
        'twist': 10,
        'mode': 'rotor',
        'nblades': n_blade,
        'diameter': 7.32 * 2,
        'radius_hub': 0.6,
        'radius': [0.61, 1.70, 4.77, 7.00],
        'chord': [0.57, 0.60, 0.50, 0.44],
        'pitch': pitch.tolist(),
        'dr': [0.545, 2.08, 2.65, 1.115 + 0.32],
        'rho': 1.19252,
        'mu': 1.81e-5,
        'coaxial': False,
    }

    solver = MyRotorSolverBeta(foil, cfg)
    T, Q, P, _ = solver.run()
    J, CT, CQ, CP, eta = solver.rotor_coeffs(T, Q, P)
    return CT, CQ, eta


def param2sol(data):
    """
    :param data: 64*4 + 1 + 4
    :return:
    """
    bs = BsplineArea(34)
    airfoils = data[:64*4].reshape(4, -1)
    secs = [bs.sampling(x)[0] for x in airfoils]
    n_blade = data[64*4]
    pitch = data[64*4 + 1:]

    return rotor_cal(secs, n_blade, pitch)


if __name__ == '__main__':

    ind = 0
    base2 = 64*4

    pitches = np.load('../rotors_dataset/pitches.npy')
    airfoils = np.load('../airfoils/conts_wgan.npy')
    perms = np.load('../rotors_dataset/rotor_keys.npy')
    res = np.load('../rotors_dataset/rotor_values.npy')

    n = len(airfoils)

    airfoil = airfoils.reshape((n, 2, 32))
    res = res.reshape((-1, 10))
    n_blade = res[:, 0]
    ct = res[:, -5]
    cq = res[:, -4]
    eta = res[:, -2]

    perms = perms[ind // base2]
    print(perms)
    data = airfoils[perms]
    data = data.reshape((-1, 32))
    pitch = pitches[ind % 64]
    n_blade = n_blade[ind]
    ct = ct[ind]
    cq = cq[ind]
    eta = eta[ind]

    x = np.hstack([data.reshape(-1), n_blade, pitch])

    ct_, cq_, eta_ = param2sol(x)

    print(abs(ct-ct_)/abs(ct))
    print(abs(cq-cq_)/abs(cq))
    print(abs(eta-eta_)/abs(eta))






