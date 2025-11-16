import ast
from dataclasses import dataclass, field
from typing import Union
from xml.sax.handler import feature_namespace_prefixes

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)

from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams


@dataclass #(frozen=True) # i don't know if this is good practise to just make it non frozen, but we have to update the tr_radius
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time
    weight_u: float = 1.0  # weight for control inputs
    # weight_d: float = 1.0  # weight for distance (can be used if we implement new objective using also distance)

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


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

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Satellite, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # Check dynamics implementation (pass this test before going further. It is not part of the final evaluation, so you can comment it out later)
        if not self.integrator.check_dynamics():
            raise ValueError("Dynamics check failed.")
        else:
            print("Dynamics check passed.")

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # commented out, because we need init_state and and goal_state for our initial guess impelmentation (defined in compute_trajectory)
        # self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        #
        # TODO: Implement SCvx algorithm or comparable
        #

        """
        for SCvx it would follow a logic similar to:
        
        initial guess interpolation
        while stopping criterion not satisfied
            convexify
            discretize
            solve convex sub problem
            update trust region
            update stopping criterion
        """

        # Main SCvx iteration loop
        for i in range(self.params.max_iterations):

            # 1. Convexify the problem around the current guess (X_bar, U_bar)
            self._convexification()

            # 2. Solve the convex subproblem
            try:
                self.problem.solve(solver=self.params.solver, verbose=self.params.verbose_solver)
            except cvx.SolverError:
                # If the solver itself crashes, shrink the trust region and try again.
                print(f"SolverError on iteration {i}: {self.params.solver} failed to solve the problem.")
                self.params.tr_radius /= self.params.alpha
                continue

            # 3. Handle non-optimal solutions (e.g., infeasible)
            if self.problem.status != "optimal":
                # If the problem was not solved to optimality, the step is invalid.
                # Shrink the trust region and try again.
                print(f"Problem not optimal on iteration {i}: status is {self.problem.status}")
                self.params.tr_radius /= self.params.alpha
                continue

            # 4. Check for convergence
            if self._check_convergence():
                # If the solution has stabilized, we are done.
                print(f"Converged after {i+1} iterations.")
                break

            # 5. Update the trust region and decide whether to accept the step
            accept_step = self._update_trust_region()

            # 6. Update the trajectory guess (X_bar) for the next iteration
            if accept_step:
                # If the step was good, update our guess with the new solution.
                self.X_bar = self.variables["X"].value
                self.U_bar = self.variables["U"].value
                self.p_bar = self.variables["p"].value
            # else: If the step was rejected, we do nothing. The loop will repeat
            # using the same X_bar but with the smaller trust region calculated
            # in _update_trust_region.

        else:
            # This 'else' belongs to the 'for' loop. It runs only if the loop
            # finishes without a 'break', meaning we ran out of iterations.
            print("Warning: SCvx algorithm did not converge within the maximum number of iterations.")

        # 7. Extract the final trajectory
        # After the loop is finished, convert the final trajectory into the required format.
        mycmds, mystates = self._extract_trajectory_from_arrays(self.X_bar, self.U_bar, self.p_bar)

        """
        self._convexification()
        try:
            error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
        except cvx.SolverError:
            print(f"SolverError: {self.params.solver} failed to solve the problem.")

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array()
        """

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        # Time discretization
        tau = np.linspace(0, 1, K)

        # Extract start and goal
        x_init = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
            ]
        )

        x_goal = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )

        # Linear interpolation between initial and goal states
        X = np.zeros((n_x, K))
        for i in range(K):
            X[:, i] = (1 - tau[i]) * x_init + tau[i] * x_goal

        # Control guess (small or zero inputs)
        U = np.zeros((n_u, K))

        # Parameter guess (e.g., final time)
        p = np.array([10.0])

        # X = np.zeros((self.satellite.n_x, K))
        # U = np.zeros((self.satellite.n_u, K))
        # p = np.zeros((self.satellite.n_p))

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.satellite.n_x, self.params.K)),
            "U": cvx.Variable((self.satellite.n_u, self.params.K)),
            "p": cvx.Variable(self.satellite.n_p),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        # dim(cvx.Parameter) <= 2
        problem_parameters = {
            "init_state": cvx.Parameter(self.satellite.n_x),
            "goal_state": cvx.Parameter(self.satellite.n_x),
            "A_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_x, self.params.K - 1)),
            "B_plus_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_u, self.params.K - 1)),
            "B_minus_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_u, self.params.K - 1)),
            "F_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_p, self.params.K - 1)),
            "r_bar": cvx.Parameter((self.satellite.n_x, self.params.K - 1)),
            "X_bar": cvx.Parameter((self.satellite.n_x, self.params.K)),
            "U_bar": cvx.Parameter((self.satellite.n_u, self.params.K)),
            "tr_radius": cvx.Parameter(shape=(), nonneg=True), # the radius is only a scalar float (shape())
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        # Write vaaraibles for better readability
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        K = self.params.K

        # Write parameters for better readability
        init_state = self.problem_parameters["init_state"]
        goal_state = self.problem_parameters["goal_state"]
        A_bar = self.problem_parameters["A_bar"]
        B_plus_bar = self.problem_parameters["B_plus_bar"]
        B_minus_bar = self.problem_parameters["B_minus_bar"]
        F_bar = self.problem_parameters["F_bar"]
        r_bar = self.problem_parameters["r_bar"]
        X_bar = self.problem_parameters["X_bar"]
        U_bar = self.problem_parameters["U_bar"]
        tr_radius = self.problem_parameters["tr_radius"]

        constraints = [
            #self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            X[:, 0] == init_state, # first state has to be init_state
            X[:, -1] == goal_state, # last state has to be goal_state
            p >= 0, # time has to be positive
            0 <= U, # control input has to be bigger/equal zero
            # Reshape F_limits to a column vector (2, 1) to allow broadcasting against U (2, 50)
            U <= np.array(self.sp.F_limits).reshape(-1, 1), # control input has to be smaller/equal F_limits
            cvx.norm(X - X_bar) <= tr_radius, # State trust region
            cvx.norm(U - U_bar) <= tr_radius, # Control trust region
        ]

        # these are the constraints for the linearized dynamics (next_state = Jacobians * current_state, inputs, time and a residual)
        for k in range(K - 1):
            # this reshaping is crucial 
            # (from the calculate_discretization we get order "F" flattend matricies, we need to reshape them to then describe the dynamic constraints)
            # also the dimension of csv.Paramter has to be <= 2.
            A_bar_k = cvx.reshape(A_bar[:, k], (self.satellite.n_x, self.satellite.n_x), order="F")
            B_plus_bar_k = cvx.reshape(B_plus_bar[:, k], (self.satellite.n_x, self.satellite.n_u), order="F")
            B_minus_bar_k = cvx.reshape(B_minus_bar[:, k], (self.satellite.n_x, self.satellite.n_u), order="F")
            F_bar_k = cvx.reshape(F_bar[:, k], (self.satellite.n_x, self.satellite.n_p), order="F")

            constraints.append(
                X[:, k + 1]
                == A_bar_k @ X[:, k]
                + B_plus_bar_k @ U[:, k + 1]
                + B_minus_bar_k @ U[:, k]
                + F_bar_k @ p
                + r_bar[:, k]
            )
        
        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        # objective = self.params.weight_p @ self.variables["p"]

        # This objective minimizes both for time (factor 10) and fuel (factor 1)
        objective = self.params.weight_p @ self.variables["p"] + self.params.weight_u * cvx.sum_squares(
            self.variables["U"]
        )

        # We could also include in the cost function the distance of the path (minimize also for the distance)
        # TODO

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)

        # The `calculate_discretization` call performs both linearization and discretization in one step.
        # It evaluates the symbolic Jacobians (from SatelliteDyn) at each point along the current trajectory guess (X_bar, U_bar)
        # and then integrates the resulting continuous-time linear system to produce the discrete-time matrices (A_bar, B_plus_bar, etc.).
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        # HINT: be aware that the matrices returned by calculate_discretization are flattened in F order (this way affect your code later when you use them)

        # All of these parameters need to be populated that the solver can check the constraints!

        # Populate init_state / goal_state parameter using the first / last state of the current guess (correctness dependent on contraints)
        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        self.problem_parameters["goal_state"].value = self.X_bar[:, -1]

        # Populate dynamics parameters
        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar
        
        # Populate remaining parameters
        self.problem_parameters["tr_radius"].value = self.params.tr_radius
        self.problem_parameters["X_bar"].value = self.X_bar
        self.problem_parameters["U_bar"].value = self.U_bar


    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        # I'm implementing the changing of cost improvement as a stopping criterion.
        # But we can and should also try other stopping criterions (see slides of excercise)

        # Calculate the non-linear cost of the previous trajectory guess (J_old)
        J_old = self.params.weight_p @ self.p_bar + self.params.weight_u * np.sum(self.U_bar**2)

        # Get the predicted cost of the new solution from the linearized problem (L_new)
        L_new = self.problem.value

        # Check if the predicted improvement is less than or equal to the stopping criterion
        predicted_improvement = J_old - L_new
        return predicted_improvement <= self.params.stop_crit

    def _update_trust_region(self) -> bool:
        """
        Update trust region radius and decide whether to accept or reject the step.
        Returns True if the step is accepted, False otherwise.
        """
        # Calculate the non-linear cost of the previous trajectory guess (J_old)
        J_old = self.params.weight_p @ self.p_bar + self.params.weight_u * np.sum(self.U_bar**2)

        # Calculate the non-linear cost of the new trajectory guess (J_new)
        J_new = (
            self.params.weight_p @ self.variables["p"].value
            + self.params.weight_u * np.sum(self.variables["U"].value ** 2)
        )

        # Get the predicted cost of the new solution from the linearized problem (L_new)
        L_new = self.problem.value

        # Calculate the actual and predicted improvement
        actual_improvement = J_old - J_new
        predicted_improvement = J_old - L_new

        # Avoid division by zero; if predicted improvement is not positive, the step is bad.
        if predicted_improvement <= 1e-4: # Use a small tolerance
            rho = -1 # Indicates a bad step
        else:
            rho = actual_improvement / predicted_improvement

        # Update the trust region radius based on the rho value
        if rho <= self.params.rho_0:
            # Very inaccurate, shrink trust region and reject the step
            self.params.tr_radius /= self.params.alpha
            accept_step = False
        elif rho < self.params.rho_1:
            # A bit inaccurate, shrink trust region but accept step
            self.params.tr_radius /= self.params.alpha
            accept_step = True
        elif rho < self.params.rho_2:
            # Quite accurate, keep trust region and accept step
            accept_step = True
        else: # rho >= self.params.rho_2
            # Very accurate, expand trust region and accept step
            self.params.tr_radius *= self.params.beta
            accept_step = True

        # Clamp the trust region radius to its min/max values
        self.params.tr_radius = np.clip(
            self.params.tr_radius, self.params.min_tr_radius, self.params.max_tr_radius
        )
        
        return accept_step

    def _extract_trajectory_from_arrays(
        self, X_bar: NDArray, U_bar: NDArray, p_bar: NDArray
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Extracts DgSampledSequence from numpy arrays and timestamps.
        """
        K = self.params.K
        final_time = p_bar[0]
        timestamps = tuple(np.linspace(0, final_time, K))

        # Commands
        F_left = U_bar[0, :]
        F_right = U_bar[1, :]
        cmds_list = [SatelliteCommands(f_l, f_r) for f_l, f_r in zip(F_left, F_right)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=timestamps, values=cmds_list)

        # States
        states = [SatelliteState(*X_bar[:, i]) for i in range(K)]
        mystates = DgSampledSequence[SatelliteState](timestamps=timestamps, values=states)
        return mycmds, mystates

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SatelliteCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 6)
        states = [SatelliteState(*v) for v in npstates]
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states)
        return mycmds, mystates
