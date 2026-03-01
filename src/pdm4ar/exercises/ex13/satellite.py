import sympy as spy

from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

import numpy as np


class SatelliteDyn:
    sg: SatelliteGeometry
    sp: SatelliteParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SatelliteGeometry, sp: SatelliteParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust_l thrust_r", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

        self.buff = 1.0 * np.sqrt(
            max(
                self.sg.l_r**2 + (self.sg.w_panel + self.sg.w_half) ** 2,
                self.sg.l_f**2 + (self.sg.w_panel + self.sg.w_half) ** 2,
                (self.sg.l_f + self.sg.l_c) ** 2,
            )
        )

        self.F_max = max(self.sp.F_limits)

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        extract the state from self.x the following way:
        0x 1y 2psi 3vx 4vy 5dpsi
        """

        _, _, psi, vx, vy, dpsi = self.x
        thrust_l, thrust_r = self.u
        t_f = self.p

        m = self.sp.m_v
        I = self.sg.Iz
        l_m = self.sg.l_m

        f = spy.zeros(self.n_x, 1)  # replace this line by computing the dynamics like in following example
        f[0] = vx
        f[1] = vy
        f[2] = dpsi
        f[3] = (1 / m) * spy.cos(psi) * (thrust_r + thrust_l)
        f[4] = (1 / m) * spy.sin(psi) * (thrust_r + thrust_l)
        f[5] = (l_m / I) * (thrust_r - thrust_l)

        f = f * t_f

        # HINT for the dynamics: SymPy is a library for symbolic mathematics.
        # Here you’ll need symbolic math, not numerical math.
        # That means you should use SymPy functions (like sympy.sin, sympy.exp, …)
        # together with symbols, instead of numerical functions from numpy or math.

        # Jacobians and matrices of the system (don't need to change)
        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
