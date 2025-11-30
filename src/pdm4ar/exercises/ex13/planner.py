import ast
from dataclasses import dataclass, field
from re import X
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)
from sympy import ordered
import numpy as np
from numpy.linalg import norm

from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex11.goal import DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable wgit reset --hardeight
    # weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time
    weight_p = 10.0
    weight_U = 1.0

    tr_radius: float = 5  # initial trust region radius
    tr_weights = {"X": 1, "U": 1, "p": 1}
    min_tr_radius: float = 1e-5  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-4  # Stopping criteria constant
    eps_r: float = 1e-4

    map_characteristic_dimension: float = 11.0

    map_edge_constr_activation_time = 5
    dock_constr_activation_time = K - 7

    use_p_jacobian = True


class Obstacle:
    def __init__(self, radius: float, start: NDArray = np.zeros(2), velocity: NDArray = np.zeros(2), is_in_dock=False):
        self.radius = radius
        self.start = start
        self.velocity = velocity
        self.is_in_dock = is_in_dock

    @staticmethod
    def from_pdm4ar(obs: PlanetParams | AsteroidParams):
        if isinstance(obs, PlanetParams):
            return Obstacle(radius=obs.radius, start=np.array(obs.center))
        elif isinstance(obs, AsteroidParams):
            return Obstacle(
                radius=obs.radius, start=np.array(obs.start), velocity=R(obs.orientation) @ np.array(obs.velocity)
            )

    def pos(self, tau: float) -> NDArray:
        return self.start + tau * self.velocity


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters

    goal: DockingTarget

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    tr_radius: float

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
        goal: DockingTarget,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp

        self.goal = goal

        # unify obstacles
        self.obstacles = [
            Obstacle.from_pdm4ar(obs) for obs in list(self.planets.values()) + list(self.asteroids.values())
        ]

        # model the docking line
        if isinstance(self.goal, DockingTarget):
            _, _, _, A1, A2, _ = self.goal.get_landing_constraint_points()

            for point in [A1, A2]:
                self.obstacles.append(Obstacle(radius=0.1, start=point, is_in_dock=True))

        self.n_obs = len(self.obstacles)

        self.params = SolverParameters()

        self.satellite = SatelliteDyn(self.sg, self.sp)

        # self.integrator = ZeroOrderHold(self.Satellite, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # if not self.integrator.check_dynamics():
        #    raise ValueError("Dynamics check failed.")
        # else:
        #    print("Dynamics check passed.")

        self.variables = self._get_variables()
        self.problem_parameters = self._get_problem_parameters()

        constraints = self._get_constraints()
        objective = self._get_objective()
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        self.tr_radius = self.params.tr_radius

        self.init_state = init_state
        self.goal_state = goal_state
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        for i in range(self.params.max_iterations):
            if self.tr_radius < self.params.min_tr_radius:
                break

            # Linearize the system about the reference trajectory
            self._convexification()
            self.J_bar = self.nonlinear_convex_cost(self.X_bar, self.U_bar, self.p_bar)

            # Solve the linearized subproblem and handle suboptimal solutions
            try:
                self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
                L_star = self.problem.value
                if self.problem.status != cvx.OPTIMAL:
                    raise cvx.SolverError
            except cvx.SolverError:
                self.tr_radius /= self.params.alpha
                continue

            X_star = self.variables["X"].value
            U_star = self.variables["U"].value
            p_star = self.variables["p"].value
            J_star = self.nonlinear_convex_cost(X_star, U_star, p_star)

            print(i, self.J_bar, J_star, L_star)

            # Check convergence
            eps = self.params.stop_crit
            max_state_change_conv_crit = norm(p_star - self.p_bar) + norm(X_star - self.X_bar, 1) < eps

            dJ_predicted = self.J_bar - L_star

            cost_improvement_conv_crit = True
            if dJ_predicted < eps:
                cost_improvement_conv_crit = dJ_predicted < self.params.eps_r * abs(self.J_bar)
            # cost_improvement_conv_crit = (
            #     True if dJ_predicted < eps else dJ_predicted < self.params.eps_r * abs(self.J_bar)
            # )

            converged = max_state_change_conv_crit or cost_improvement_conv_crit

            # Update trust region and accept or reject the solution
            accepted = self._update_trust_region(self.J_bar, J_star, L_star) or i == 0

            if accepted:
                self.X_bar = X_star
                self.U_bar = U_star
                self.p_bar = p_star

            if converged:
                break

        mycmds, mystates = self._postprocess_solution(self.X_bar, self.U_bar, self.p_bar)

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u

        X0 = self.init_state.as_ndarray()
        Xf = self.goal_state.as_ndarray()

        X = np.zeros((n_x, K))
        ts = np.linspace(0, 1.0, K, endpoint=True)
        for k in range(K):
            X[:, k] = (1 - ts[k]) * X0 + ts[k] * Xf
        U = np.zeros((n_u, K))

        m = self.satellite.sp.m_v
        F_max = self.satellite.F_max
        t_f_guess = 4 * np.sqrt(m * norm(X0[:2] - Xf[:2]) / F_max)
        p = np.array([t_f_guess])
        p = np.array([10.0])

        return X, U, p

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        K = self.params.K

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        variables = {
            "X": cvx.Variable((n_x, K)),
            "U": cvx.Variable((n_u, K)),
            "p": cvx.Variable(n_p),
            # slack
            "nu": cvx.Variable((n_x, K - 1)),
            "nu_ic": cvx.Variable(n_x),
            "nu_tc": cvx.Variable(n_x),
        }

        if self.obstacles:
            variables["nu_s"] = cvx.Variable((self.n_obs, K))

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        K = self.params.K

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        n_obs = self.n_obs

        problem_parameters = {
            "init_state": cvx.Parameter(n_x),
            "goal_state": cvx.Parameter(n_x),
            "X_bar": cvx.Parameter((n_x, K)),
            "U_bar": cvx.Parameter((n_u, K)),
            "p_bar": cvx.Parameter(n_p),
            "A_bar": cvx.Parameter((n_x * n_x, K - 1)),
            "B_minus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "B_plus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "F_bar": cvx.Parameter((n_x * n_p, K - 1)),
            "r_bar": cvx.Parameter((n_x, K - 1)),
            "tr_radius": cvx.Parameter(),
        }

        if self.obstacles:
            problem_parameters["C_bar"] = cvx.Parameter((n_obs * n_x, K))
            problem_parameters["G_bar"] = cvx.Parameter((n_obs * n_p, K))
            problem_parameters["r_prim_bar"] = cvx.Parameter((n_obs, K))

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        buff = self.satellite.buff

        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        nu = self.variables["nu"]
        nu_ic = self.variables["nu_ic"]
        nu_tc = self.variables["nu_tc"]

        init_state = self.problem_parameters["init_state"]
        goal_state = self.problem_parameters["goal_state"]

        X_bar = self.problem_parameters["X_bar"]
        U_bar = self.problem_parameters["U_bar"]
        p_bar = self.problem_parameters["p_bar"]

        A_bar = self.problem_parameters["A_bar"]
        B_plus_bar = self.problem_parameters["B_plus_bar"]
        B_minus_bar = self.problem_parameters["B_minus_bar"]
        F_bar = self.problem_parameters["F_bar"]
        r_bar = self.problem_parameters["r_bar"]

        tr_radius = self.problem_parameters["tr_radius"]

        # boundary conditions
        constraints = [
            X[:, 0] + nu_ic == init_state,
            X[:, -1] + nu_tc == goal_state,
            U[:, 0] == np.zeros(n_u),
            U[:, -1] == np.zeros(n_u),
            p[0] >= 0,
        ]

        # dynamics constraint
        for k in range(K - 1):
            A_bar_k = cvx.reshape(A_bar[:, k], (n_x, n_x), order="F")
            B_minus_bar_k = cvx.reshape(B_minus_bar[:, k], (n_x, n_u), order="F")
            B_plus_bar_k = cvx.reshape(B_plus_bar[:, k], (n_x, n_u), order="F")
            F_bar_k = cvx.reshape(F_bar[:, k], (n_x, n_p), order="F")

            constraints.append(
                X[:, k + 1]
                == A_bar_k @ X[:, k]
                + B_minus_bar_k @ U[:, k]
                + B_plus_bar_k @ U[:, k + 1]
                + F_bar_k @ p
                + r_bar[:, k]
                + nu[:, k]
            )

        # convex path constraints
        F_max = self.satellite.sp.F_limits[1]
        D_max = self.params.map_characteristic_dimension - buff
        constraints.extend([cvx.norm(U[:, k], "inf") <= F_max for k in range(K)])
        constraints.extend(
            [cvx.norm(X[:2, k], "inf") <= D_max for k in range(self.params.map_edge_constr_activation_time, K)]
        )

        # nonconvex path constraints
        if self.obstacles:
            C_bar = self.problem_parameters["C_bar"]
            G_bar = self.problem_parameters["G_bar"]
            r_prim_bar = self.problem_parameters["r_prim_bar"]
            nu_s = self.variables["nu_s"]

            for k in range(K):
                for j, obs in enumerate(self.obstacles):
                    if obs.is_in_dock and k >= self.params.dock_constr_activation_time + 1:
                        continue

                    C_bar_j_k = cvx.reshape(C_bar[j * n_x : (j + 1) * n_x, k], (1, n_x))
                    G_bar_j_k = G_bar[j, k]
                    r_prim_bar_j_k = r_prim_bar[j, k]

                    constraints.append(C_bar_j_k @ X[:, k] + G_bar_j_k * p[0] + r_prim_bar_j_k <= nu_s[j, k])

        # trust region constraint
        w_X, w_U, w_p = self.params.tr_weights.values()
        for k in range(K):
            constraints.append(
                w_X * cvx.norm(X[:, k] - X_bar[:, k], "inf")
                + w_U * cvx.norm(U[:, k] - U_bar[:, k], "inf")
                + w_p * cvx.norm(p - p_bar, "inf")
                <= tr_radius
            )

        # docking constraints
        if isinstance(self.goal, DockingTarget):
            A, B, C, A1, A2, _ = self.goal.get_landing_constraint_points_offset()

            dock_constr_activation_time = self.params.dock_constr_activation_time

            range_dock_constr = range(dock_constr_activation_time, K)

            # returns the constraint of a halfplane defined by points P1 and P2.
            # The direction of the halfplane is given by P_ref (i.e. P_ref lies *within* the halfplane)
            # and the constraint forces X to be *outside* of the halfplane
            def _halfplane_constraint(P1, P2, P_ref, rng, buff=0):
                if np.cross(P_ref - P1, P2 - P1) > 0:
                    return [_cvx_cross2d(X[:2, k] - P1, P2 - P1) + buff <= 0 for k in rng]
                else:
                    return [_cvx_cross2d(X[:2, k] - P1, P2 - P1) - buff >= 0 for k in rng]

            constraint_lists = [
                # cone constraint
                _halfplane_constraint(A, B, A1, range_dock_constr),
                _halfplane_constraint(A, C, A2, range_dock_constr),
            ]

            for constraint in constraint_lists:
                constraints.extend(constraint)

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        K = self.params.K
        n_x = self.satellite.n_x

        U = self.variables["U"]
        p = self.variables["p"]
        nu = cvx.hstack([self.variables["nu"], np.zeros((n_x, 1))])
        nu_ic = self.variables["nu_ic"]
        nu_tc = self.variables["nu_tc"]
        nu_s = np.zeros((1, K)) if "nu_s" not in self.variables else self.variables["nu_s"]

        lambda_nu = self.params.lambda_nu
        w_p = self.params.weight_p
        w_U = self.params.weight_U

        def gamma(k):
            run_cost = w_U * cvx.norm(U[:, k], 1) + lambda_nu * (cvx.norm(nu_s[:, k], 1) + cvx.norm(nu[:, k], 1))
            return run_cost

        objective = w_p * cvx.norm(p[0]) + lambda_nu * (cvx.norm(nu_ic, 1) + cvx.norm(nu_tc, 1)) + trapz(gamma, K)

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )
        # Solver Parameters

        # HINT: be aware that the matrices returned by calculate_discretization are flattened in F order (this way affect your code later when you use them)

        self.problem_parameters["init_state"].value = self.init_state.as_ndarray()
        self.problem_parameters["goal_state"].value = self.goal_state.as_ndarray()

        self.problem_parameters["X_bar"].value = self.X_bar
        self.problem_parameters["U_bar"].value = self.U_bar
        self.problem_parameters["p_bar"].value = self.p_bar

        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar

        if self.obstacles:
            _, C_bar, G_bar, r_prim_bar = self.discrete_nonconvex_constraints(self.X_bar, self.p_bar)

            self.problem_parameters["C_bar"].value = C_bar
            self.problem_parameters["G_bar"].value = G_bar
            self.problem_parameters["r_prim_bar"].value = r_prim_bar

        self.problem_parameters["tr_radius"].value = self.tr_radius

    def _update_trust_region(self, J_bar: float, J_star: float, L_star: float) -> bool:

        dJ_actual = J_bar - J_star
        dJ_predicted = J_bar - L_star

        rho = -1.0 if dJ_predicted < 1e-9 else dJ_actual / dJ_predicted

        accepted = True

        if rho < self.params.rho_0:
            self.tr_radius /= self.params.alpha
            accepted = False
        elif rho < self.params.rho_1:
            self.tr_radius /= self.params.alpha
        elif rho >= self.params.rho_2:
            self.tr_radius *= self.params.beta

        return accepted

    def _postprocess_solution(self, X: NDArray, U: NDArray, p: NDArray):
        K = self.params.K
        t_f = p[0]
        ts = np.linspace(0, t_f, K, endpoint=True)

        mycmds = DgSampledSequence[SatelliteCommands](
            timestamps=ts, values=[SatelliteCommands(F_left=U[0, k], F_right=U[1, k]) for k in range(K)]
        )

        mystates = DgSampledSequence[SatelliteState](
            timestamps=ts, values=[SatelliteState.from_array(X[:, k]) for k in range(K)]
        )

        return mycmds, mystates

    def discrete_nonconvex_constraints(self, X: NDArray, p: NDArray):
        K = self.params.K

        n_x = self.satellite.n_x
        n_p = self.satellite.n_p

        obstacles = self.obstacles
        n_obs = self.n_obs
        buff = self.satellite.buff

        s = np.zeros((n_obs, K))
        C = np.zeros((n_obs * n_x, K))
        G = np.zeros((n_obs * n_p, K))
        r = np.zeros((n_obs, K))

        for k in range(K):
            t = k / (K - 1)
            tau_k = p[0] * t
            p_bar_k = X[:2, k].flatten()  # position component of the satellite's state

            for j, obs in enumerate(obstacles):
                if obs.is_in_dock and k >= self.params.dock_constr_activation_time + 1:
                    continue
                dist_vec = p_bar_k - obs.pos(tau_k)

                s_j_k = (obs.radius + buff) ** 2 - norm(dist_vec) ** 2
                s[j, k] = s_j_k

                C[j * n_x : j * n_x + 2, k] = -2 * dist_vec
                C_j_k = np.reshape(C[j * n_x : (j + 1) * n_x, k], (1, n_x), order="F")

                r[j, k] = s[j, k] - (C_j_k @ X[:, k]).item()

                if self.params.use_p_jacobian:
                    G[j, k] = 2 * t * np.dot(dist_vec, obs.velocity)
                    r[j, k] -= G[j, k] * p[0]

        return s, C, G, r

    def nonlinear_convex_cost(self, X: NDArray, U: NDArray, p: NDArray):

        K = self.params.K
        lambda_nu = self.params.lambda_nu
        w_p = self.params.weight_p
        w_U = self.params.weight_U

        X0 = self.problem_parameters["init_state"].value
        Xf = self.problem_parameters["goal_state"].value

        flow_map = self.integrator.integrate_nonlinear_piecewise(X, U, p)

        defect = np.zeros((self.satellite.n_x, K))
        defect[:, :-1] = X[:, 1:] - flow_map[:, 1:]

        s_plus = np.maximum(0, self.discrete_nonconvex_constraints(X, p)[0])

        def gamma(k):
            run_cost = w_U * norm(U[:, k], 1) + lambda_nu * (norm(defect[:, k], 1) + norm(s_plus[:, k], 1))
            return run_cost

        J = w_p * p[0] + lambda_nu * (norm(X[:, 0] - X0, 1) + norm(X[:, -1] - Xf, 1)) + trapz(gamma, K)

        return J


def R(th: float):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def trapz(func, N):
    dt = 1.0 / (N - 1)
    vals = [func(k) for k in range(N)]
    return dt / 2 * (vals[0] + vals[-1] + 2 * sum(vals[1:-1]))


def _cvx_cross2d(cvx_2d_vec, np_2d_vec):
    return cvx_2d_vec[0] * np_2d_vec[1] - cvx_2d_vec[1] * np_2d_vec[0]
