import ast
from dataclasses import dataclass, field
from typing import Union
from xml.sax.handler import feature_namespace_prefixes

import matplotlib.pyplot as plt
import numpy as np
from click import clear
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


@dataclass  # (frozen=True) # i don't know if this is good practise to just make it non frozen, but we have to update the tr_radius
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    #solver: str = "SCS"
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time
    weight_u: float = 100.0  # weight for control inputs
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
    relative_stop_crit: float = 1e-2  # Stopping criteria for relative cost improvement


@dataclass
class SCvxIterationLog:
    """
    Data structure to store metrics for a single SCvx iteration.
    """
    iteration: int
    tr_radius: float
    
    # Merits and Costs
    J_bar: float  # Nonlinear merit of reference (old)
    L_star: float # Linearized merit of candidate (new)
    J_star: float # Nonlinear merit of candidate (new)
    
    # Improvements and Ratio
    pred_improv: float
    act_improv: float
    rho: float
    
    # Status
    accepted: bool
    status: str # Solver status
    
    # Slacks (L1 norms)
    norm_nu: float
    norm_nu_ic: float
    norm_nu_tc: float
    norm_nu_s: float
    
    # Convergence Metrics
    traj_change: float
    rel_improv: float
    
    # Variables (Optional: keep copies if needed for plotting)
    X: NDArray | None = None
    U: NDArray | None = None
    p: NDArray | None = None


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
    
    history: list[SCvxIterationLog]

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
        
        self.history = []

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
        # --- FIX: Reset trust region radius for each new planning request ---
        # The trust region shrinks during convergence. If we don't reset it for a new plan,
        # the solver starts with an overly restrictive search space, leading to numerical issues.
        self.params.tr_radius = 5.0  # Reset to initial default value
        
        self.history = [] # Reset history
        self._print_iteration_log_header()

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

            # # ... inside compute_trajectory loop ...

            # # 1. Calculate Merit Components for X_bar
            # merit_old = self._calculate_nonlinear_merit(self.X_bar, self.U_bar, self.p_bar)
            # # Note: You might need to modify _calculate_nonlinear_merit to return components (fuel, time, dynamics, collision)
            # # Or just manually calculate them here for debugging:

            # val_time = (self.params.weight_p @ self.p_bar).item()
            # val_fuel = self.params.weight_u * np.sum(self.U_bar**2) * (1.0/(self.params.K-1))

            # # Calculate Defects (Dynamics) for X_bar
            # X_nl = self.integrator.integrate_nonlinear_piecewise(self.X_bar, self.U_bar, self.p_bar)
            # val_dyn_defect = self.params.lambda_nu * np.sum(np.abs(self.X_bar[:, 1:] - X_nl[:, 1:])) * (1.0/(self.params.K-1))

            # # Calculate Collision Violation for X_bar
            # # COPY your squared logic from _calculate_nonlinear_merit here
            # # val_coll = ... 

            # print(f"--- COST MISMATCH DEBUG ---")
            # print(f"Merit (Ref): Time={val_time:.4e}, Fuel={val_fuel:.4e}, Dyn={val_dyn_defect:.4e}") # Add val_coll

            # # 2. Calculate Solver Cost Components for X_bar (Theoretical)
            # # If the solver stayed at X_bar, what would the cost be?
            # # It should be identical to above.

            # # 3. Compare with Solver Solution (L_star) components
            # p_new = self.variables["p"].value
            # u_new = self.variables["U"].value
            # nu_new = self.variables["nu"].value
            # # Safely get planet slack (if any)
            # if "nu_s" in self.variables:
            #     nu_s_new = self.variables["nu_s"].value
            #     sum_nu_s = np.sum(nu_s_new)
            # else:
            #     sum_nu_s = 0.0

            # # Safely get asteroid slack (if any)
            # if "nu_s_asteroids" in self.variables:
            #     nu_s_ast_new = self.variables["nu_s_asteroids"].value
            #     sum_nu_s_ast = np.sum(nu_s_ast_new)
            # else:
            #     sum_nu_s_ast = 0.0

            # sol_time = (self.params.weight_p @ p_new).item()
            # sol_fuel = self.params.weight_u * np.sum(u_new**2) * (1.0/(self.params.K-1))
            # sol_dyn = self.params.lambda_nu * np.sum(np.abs(nu_new)) * (1.0/(self.params.K-1))
            # sol_coll = self.params.lambda_nu * np.sum(nu_s_new) * (1.0/(self.params.K-1))

            # print(f"Solver (New): Time={sol_time:.4e}, Fuel={sol_fuel:.4e}, Dyn={sol_dyn:.4e}, Coll={sol_coll:.4e}")
            # print(f"Total L_new: {self.problem.value:.4e}")
            # print(f"---------------------------")

            # 4. Check for convergence (after a grace period of a few iterations)
            # Note: We check convergence BEFORE updating trust region, but we might want to log it first.
            # Ideally, we calculate everything, log it, then check convergence.
            
            # 5. Update the trust region and decide whether to accept the step
            accept_step, metrics = self._update_trust_region()

            # --- Logging ---
            # Calculate slack norms for logging
            norm_nu = np.linalg.norm(self.variables["nu"].value, 1)
            norm_nu_ic = np.linalg.norm(self.variables["nu_ic"].value, 1)
            norm_nu_tc = np.linalg.norm(self.variables["nu_tc"].value, 1)
            norm_nu_s = 0.0
            if "nu_s" in self.variables and self.variables["nu_s"].value is not None:
                 norm_nu_s += np.sum(self.variables["nu_s"].value)
            if "nu_s_asteroids" in self.variables and self.variables["nu_s_asteroids"].value is not None:
                 norm_nu_s += np.sum(self.variables["nu_s_asteroids"].value)
            
            # Calculate Convergence Metrics (for logging only)
            # 1. Trajectory Change: ||p - p_bar|| + max_k ||x_k - x_bar_k||
            p_star = self.variables["p"].value
            X_star = self.variables["X"].value
            
            p_change = np.linalg.norm(p_star - self.p_bar)
            max_x_change = 0.0
            for k in range(self.params.K):
                max_x_change = max(max_x_change, np.linalg.norm(X_star[:, k] - self.X_bar[:, k]))
            traj_change = p_change + max_x_change
            
            # 2. Relative Improvement
            j_bar = metrics["J_bar"]
            if j_bar > 1e-9:
                rel_improv = metrics["pred_improv"] / j_bar
            else:
                rel_improv = 0.0

            log_entry = SCvxIterationLog(
                iteration=i,
                tr_radius=self.params.tr_radius, # Note: this is the UPDATED radius
                J_bar=metrics["J_bar"],
                L_star=metrics["L_star"],
                J_star=metrics["J_star"],
                pred_improv=metrics["pred_improv"],
                act_improv=metrics["act_improv"],
                rho=metrics["rho"],
                accepted=accept_step,
                status=self.problem.status,
                norm_nu=norm_nu,
                norm_nu_ic=norm_nu_ic,
                norm_nu_tc=norm_nu_tc,
                norm_nu_s=norm_nu_s,
                traj_change=traj_change,
                rel_improv=rel_improv,
                # Store copies of variables if needed for plotting (heavy on memory?)
                X=self.variables["X"].value.copy()
                # U=self.variables["U"].value.copy(),
                # p=self.variables["p"].value.copy()
            )
            self.history.append(log_entry)
            self._print_iteration_log(log_entry)

            # Check convergence
            if i > 1 and self._check_convergence():
                print(f"Converged after {i+1} iterations.")
                self._plot_convergence() # Plot results
                break

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
            self._plot_convergence() # Plot partial results

        # 7. Extract the final trajectory
        # After the loop is finished, convert the final trajectory into the required format.
        print("X_bar: ", self.X_bar, "U_bar: ", self.U_bar, "p_bar: ", self.p_bar)

        # ================= ADD THIS BLOCK =================
        # Check for "Soft Failure" (Collision Slack)
        if "nu_s_asteroids" in self.variables and self.variables["nu_s_asteroids"].value is not None:
            max_ast_slack = np.max(self.variables["nu_s_asteroids"].value)
            if max_ast_slack > 1e-3:
                print(f"WARNING: Trajectory is INFEASIBLE. Max Asteroid Slack: {max_ast_slack:.6e}")
        # ==================================================

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
    
    def _print_iteration_log_header(self):
        header = (
            f"{'Iter':<5} | {'Radius':<8} | {'J_bar':<10} | {'L_star':<10} | "
            f"{'J_star':<10} | {'Imp(P)':<10} | {'Imp(A)':<10} | {'Rho':<8} | "
            f"{'Acc':<3} | {'|nu|':<9} | {'|nu_ic|':<9} | {'|nu_tc|':<9} | {'|nu_s|':<9} | {'dTraj':<9} | {'dCost':<9}"
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))

    def _print_iteration_log(self, log: SCvxIterationLog):
        print(
            f"{log.iteration:<5} | {log.tr_radius:<8.2e} | {log.J_bar:<10.4e} | {log.L_star:<10.4e} | "
            f"{log.J_star:<10.4e} | {log.pred_improv:<10.2e} | {log.act_improv:<10.2e} | {log.rho:<8.2f} | "
            f"{'Y' if log.accepted else 'N':<3} | {log.norm_nu:<9.2e} | {log.norm_nu_ic:<9.2e} | {log.norm_nu_tc:<9.2e} | {log.norm_nu_s:<9.2e} | "
            f"{log.traj_change:<9.2e} | {log.rel_improv:<9.2e}"
        )
    
    def _plot_convergence(self):
        if not self.history:
            return

        iterations = [log.iteration for log in self.history]
        
        # 1. Merit and Cost
        J_vals = [log.J_bar for log in self.history]
        J_star_vals = [log.J_star for log in self.history]
        L_star_vals = [log.L_star for log in self.history]
        
        # 2. Radius
        radii = [log.tr_radius for log in self.history]
        
        # 3. Slacks
        nu_vals = [log.norm_nu for log in self.history]
        nu_ic_vals = [log.norm_nu_ic for log in self.history]
        nu_tc_vals = [log.norm_nu_tc for log in self.history]
        nu_s_vals = [log.norm_nu_s for log in self.history]
        
        # 4. Convergence Metrics
        traj_change_vals = [log.traj_change for log in self.history]
        rel_improv_vals = [max(1e-16, log.rel_improv) for log in self.history] # Avoid log(0)

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot Cost [0, 0]
        axs[0, 0].plot(iterations, J_vals, 'b-o', label='J_bar (Ref)')
        axs[0, 0].plot(iterations, J_star_vals, 'g-x', label='J_star (Cand)')
        axs[0, 0].plot(iterations, L_star_vals, 'r--', label='L_star (Lin)')
        axs[0, 0].set_title('Merit Function Evolution')
        axs[0, 0].set_yscale('log')
        axs[0, 0].legend(fontsize='small')
        axs[0, 0].grid(True)
        
        # Plot Radius [0, 1]
        axs[0, 1].plot(iterations, radii, 'r-x')
        axs[0, 1].set_title('Trust Region Radius')
        axs[0, 1].set_yscale('log')
        axs[0, 1].grid(True)
        
        # Plot Rho [0, 2]
        rhos = [log.rho for log in self.history]
        axs[0, 2].plot(iterations, rhos, 'k-d')
        axs[0, 2].axhline(y=0, color='r', linestyle='--')
        axs[0, 2].axhline(y=self.params.rho_0, color='y', linestyle=':')
        axs[0, 2].axhline(y=self.params.rho_1, color='g', linestyle=':')
        axs[0, 2].axhline(y=self.params.rho_2, color='g', linestyle='--')
        axs[0, 2].set_title('Ratio (Rho)')
        # axs[0, 2].set_ylim(-0.5, 2.0) # Clip y-axis for readability
        axs[0, 2].grid(True)
        
        # Plot Slacks [1, 0]
        axs[1, 0].plot(iterations, nu_vals, 'g-^', label='||nu|| (Dyn)')
        axs[1, 0].plot(iterations, nu_ic_vals, 'c-s', label='||nu_ic|| (Init)')
        axs[1, 0].plot(iterations, nu_tc_vals, 'y-d', label='||nu_tc|| (Term)')
        axs[1, 0].plot(iterations, nu_s_vals, 'm-v', label='||nu_s|| (Coll)')
        axs[1, 0].set_title('Slack Variables (L1 Norm)')
        axs[1, 0].set_yscale('log')
        axs[1, 0].legend(fontsize='small')
        axs[1, 0].grid(True)
        
        # Plot Convergence Criteria [1, 1]
        axs[1, 1].plot(iterations, traj_change_vals, 'b-s', label='Traj Change')
        axs[1, 1].plot(iterations, rel_improv_vals, 'r-o', label='Rel Improv')
        axs[1, 1].axhline(y=self.params.stop_crit, color='b', linestyle='--', label='Traj Thresh')
        axs[1, 1].axhline(y=self.params.relative_stop_crit, color='r', linestyle='--', label='Cost Thresh')
        axs[1, 1].set_title('Convergence Criteria')
        axs[1, 1].set_yscale('log')
        axs[1, 1].legend(fontsize='small')
        axs[1, 1].grid(True)
        
        # Plot Trajectories [1, 2]
        # Iterate through history and plot X trajectories
        # X shape is (n_x, K). Position is index 0 and 1 (x, y).
        
        total_iters = len(self.history)
        # We want: Start (0), End (N-1), and max 3 intermediates.
        # Total plots = 2 + 3 = 5.
        # If total_iters <= 5, plot all.
        # Else, we need a stride.
        if total_iters <= 5:
            plot_step = 1
        else:
            # We want to pick 3 indices between 1 and N-2.
            # Example: N=10. Indices: 0, 1..8, 9.
            # We can just divide the range into 4 segments?
            # Simple approach: stride = total // 4
            plot_step = total_iters // 4

        for idx, log in enumerate(self.history):
            if log.X is None:
                continue
            
            X = log.X
            x_coords = X[0, :]
            y_coords = X[1, :]
            
            if idx == 0:
                # Initial Guess / First Iteration
                axs[1, 2].plot(x_coords, y_coords, 'r--', label='Iter 0', alpha=0.8)
            elif idx == total_iters - 1:
                # Final Iteration
                axs[1, 2].plot(x_coords, y_coords, 'g-', label=f'Final (Iter {log.iteration})', linewidth=2)
            elif idx % plot_step == 0:
                # Intermediate Iterations
                axs[1, 2].plot(x_coords, y_coords, 'b-', alpha=0.2)

        # Plot start and goal
        # Use init_state and goal_state from the log or params if available, 
        # but strictly we have them in the planner object.
        # We can access self.init_state (SatelliteState) and self.goal_state (DynObstacleState)
        axs[1, 2].plot(self.init_state.x, self.init_state.y, 'ko', label='Start')
        axs[1, 2].plot(self.goal_state.x, self.goal_state.y, 'rx', label='Goal')

        axs[1, 2].set_title('Trajectory Evolution (X vs Y)')
        axs[1, 2].set_xlabel('X [m]')
        axs[1, 2].set_ylabel('Y [m]')
        axs[1, 2].legend(fontsize='small')
        axs[1, 2].grid(True)
        axs[1, 2].axis('equal')
        
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs("plots", exist_ok=True)
        # Use timestamp or unique ID to avoid overwriting if run multiple times rapidly? 
        # For now, iteration count is decent.
        filename = f"plots/convergence_iter_{self.history[-1].iteration}.png"
        plt.savefig(filename)
        print(f"Convergence plot saved to {filename}")
        plt.close(fig)

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
        # U = np.ones((n_u, K)) * 0.1

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
        K = self.params.K

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        n_planets = len(self.planets)
        n_asteroids = len(self.asteroids)

        variables = {
            "X": cvx.Variable((n_x, K)),
            "U": cvx.Variable((n_u, K)),
            "p": cvx.Variable(n_p),
            # Dynamic virtual control (Eq 47a)
            "nu": cvx.Variable((n_x, K - 1)),
            # Initial Condition virtual control (Eq 47c)
            "nu_ic": cvx.Variable(n_x), 
            # Terminal Condition virtual control (Eq 47d)
            "nu_tc": cvx.Variable(n_x),
        }

        if n_planets > 0:
            variables["nu_s"] = cvx.Variable(n_planets * K, nonneg=True) # Slack for planet constraints

        if n_asteroids > 0:
            variables["nu_s_asteroids"] = cvx.Variable(n_asteroids * K, nonneg=True) # Slack for asteroid constraints

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        K = self.params.K

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        n_planets = len(self.planets)
        n_asteroids = len(self.asteroids)

        # dim(cvx.Parameter) <= 2
        problem_parameters = {
            "init_state": cvx.Parameter(n_x),
            "goal_state": cvx.Parameter(n_x),
            "X_bar": cvx.Parameter((n_x, K)),
            "U_bar": cvx.Parameter((n_u, K)),
            "p_bar": cvx.Parameter(n_p),
            "A_bar": cvx.Parameter((n_x * n_x, K - 1)),
            "B_plus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "B_minus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "F_bar": cvx.Parameter((n_x * n_p, K - 1)),
            "r_bar": cvx.Parameter((n_x, K - 1)),
            "tr_radius": cvx.Parameter(),  # the radius is only a scalar float (shape())
        }

        if n_planets > 0:
            problem_parameters["planet_C"] = cvx.Parameter((n_planets * K, 2))
            problem_parameters["planet_r_prime"] = cvx.Parameter(n_planets * K)

        if n_asteroids > 0:
            problem_parameters["asteroid_C"] = cvx.Parameter((n_asteroids * K, 2))
            problem_parameters["asteroid_r_prime"] = cvx.Parameter(n_asteroids * K)

        # These parameters are flattened to account for cvxpy's 2D parameter limit.
        # The index 'idx' for these flattened arrays maps to (planet_idx, time_step_k)
        # using the formula: idx = planet_idx * self.params.K + time_step_k.
        #
        # For 'planet_C', the second dimension (column 0 and 1) represents the x and y
        # components of the C vector (i.e., C_k = [C_x, C_y]).

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

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
            #X[:, 0] == init_state,
            #X[:, -1] == goal_state,
            X[:, 0] + nu_ic == init_state,      # Initial condition with slack
            X[:, -1] + nu_tc == goal_state,     # Terminal condition with slack
            U[:, 0] == np.array([0, 0]),
            U[:, -1] == np.array([0, 0]),
            p[0] >= 0,
        ]

        # dynamics constraints
        for k in range(K - 1):
            A_bar_k = cvx.reshape(A_bar[:, k], (n_x, n_x), order="F")
            B_plus_bar_k = cvx.reshape(B_plus_bar[:, k], (n_x, n_u), order="F")
            B_minus_bar_k = cvx.reshape(B_minus_bar[:, k], (n_x, n_u), order="F")
            F_bar_k = cvx.reshape(F_bar[:, k], (n_x, n_p), order="F")

            constraints.append(
                X[:, k + 1]
                == A_bar_k @ X[:, k]
                + B_plus_bar_k @ U[:, k + 1]
                + B_minus_bar_k @ U[:, k]
                + F_bar_k @ p
                + r_bar[:, k]
                + nu[:, k]
            )

        # convex path constraints
        F_max = self.satellite.sp.F_limits[1]
        for k in range(K):
            constraints.append(cvx.norm(U[:, k], "inf") <= F_max)

        # Per–time-step trust region constraints (51g)
        for k in range(K):
            constraints.append(
                cvx.norm(X[:, k] - X_bar[:, k], "inf")
                + cvx.norm(U[:, k] - U_bar[:, k], "inf")
                + cvx.norm(p - p_bar, "inf")
                <= tr_radius
            )


        # Planet collision avoidance constraints:
        # The original non-convex constraint for a planet p at timestep k is:
        # ||X[:2, k] - p_center||^2 >= p_radius^2
        #
        # This can be written as h(x_k) <= 0, where h(x_k) = p_radius^2 - ||x_k - p_center||^2.
        # We linearize this non-convex constraint around the previous trajectory point x_bar_k.
        # The linearized constraint is: h(x_bar_k) + grad(h(x_bar_k))^T * (x_k - x_bar_k) <= 0.
        #
        # After rearranging, this gives a linear constraint of the form:
        # C_k^T * x_k + r'_k <= 0
        #
        # where:
        # C_k = grad(h(x_bar_k)) = -2 * (x_bar_k - p_center). This is stored in `planet_C`.
        # r'_k is a constant term derived from the linearization. Stored in `planet_r_prime`.
        if len(self.planets) > 0:
            planet_C = self.problem_parameters["planet_C"]
            planet_r_prime = self.problem_parameters["planet_r_prime"]
            idx = 0
            for _ in self.planets:
                for k in range(K):
                    # The C_k vector from `planet_C[idx, :]` is a 1x2 row vector.
                    # X[:2, k] is the 2x1 position vector [x, y]^T.
                    # Their product gives the scalar C_k^T * x_k.
                    constraints.append(planet_C[idx, :] @ X[:2, k] + planet_r_prime[idx] <= self.variables["nu_s"][idx])
                    idx += 1

        # Asteroid collision avoidance constraints:
        # The original non-convex constraint for an asteroid 'a' at timestep k is:
        # ||X[:2, k] - a_center_k||^2 >= a_radius^2
        # where a_center_k is the predicted position of the asteroid at time k.
        #
        # This can be written as h(x_k) <= 0, where h(x_k) = a_radius^2 - ||x_k - a_center_k||^2.
        # We linearize this non-convex constraint around the previous trajectory point x_bar_k.
        # The linearized constraint is: h(x_bar_k) + grad(h(x_bar_k))^T * (x_k - x_bar_k) <= 0.
        #
        # After rearranging, this gives a linear constraint of the form:
        # C_k^T * x_k + r'_k <= 0
        #
        # where:
        # C_k = grad(h(x_bar_k)) = -2 * (x_bar_k - a_center_k). This is stored in `asteroid_C`.
        # r'_k is a constant term derived from the linearization. Stored in `asteroid_r_prime`.
        # Note that a_center_k is time-dependent.
        if len(self.asteroids) > 0:
            asteroid_C = self.problem_parameters["asteroid_C"]
            asteroid_r_prime = self.problem_parameters["asteroid_r_prime"]
            idx = 0
            for _ in self.asteroids:
                for k in range(K):
                    # The C_k vector from `asteroid_C[idx, :]` is a 1x2 row vector.
                    # X[:2, k] is the 2x1 position vector [x, y]^T.
                    # Their product gives the scalar C_k^T * x_k.
                    constraints.append(
                        asteroid_C[idx, :] @ X[:2, k] + asteroid_r_prime[idx] <= self.variables["nu_s_asteroids"][idx]
                    )
                    idx += 1

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        # objective = self.params.weight_p @ self.variables["p"]

        # This objective minimizes both for time (factor 10) and fuel (factor 1)
        # objective = self.params.weight_p @ self.variables["p"] + self.params.weight_u * cvx.sum_squares(
        #     self.variables["U"]
        # )
        """
        objective = (
            self.params.weight_p @ self.variables["p"]
            + self.params.weight_u * cvx.sum_squares(self.variables["U"])
            + self.params.lambda_nu * cvx.norm(self.variables["nu"], 1)
            # + self.params.lambda_nu * cvx.norm(self.variables["nu_tc"], 1)
            # + self.params.lambda_nu * cvx.norm(self.variables["nu_s_k"], 1)
        )
        if len(self.planets) > 0:
            objective += self.params.lambda_nu * cvx.sum(self.variables["nu_s"])

        if len(self.asteroids) > 0:
            objective += self.params.lambda_nu * cvx.sum(self.variables["nu_s_asteroids"])

        # We could also include in the cost function the distance of the path (minimize also for the distance)
        # TODO

        return cvx.Minimize(objective)
        """
        # Variables
        U = self.variables["U"]
        p = self.variables["p"]
        nu = self.variables["nu"]     # Shape (n_x, K-1)
        nu_ic = self.variables["nu_ic"]
        nu_tc = self.variables["nu_tc"]
        
        K = self.params.K
        dt_norm = 1.0 / (K - 1)
        lambda_val = self.params.lambda_nu

        # --- 1. Augmented Terminal Cost ---
        J_terminal = self.params.weight_p @ p
        penalty_boundary = lambda_val * (cvx.norm(nu_ic, 1) + cvx.norm(nu_tc, 1))

        # --- 2. Augmented Running Cost (Trapezoidal) ---
        integral_sum = 0
        
        for k in range(K):
            # Weight: 0.5 for endpoints, 1.0 otherwise
            weight = 0.5 if (k == 0 or k == K - 1) else 1.0
            
            # A. Fuel Cost
            cost_fuel = self.params.weight_u * cvx.sum_squares(U[:, k])
            
            # B. Dynamics Slack (nu)
            # nu is defined for k=0 to K-2. For k=K-1, slack is 0 (per paper text).
            if k < K - 1:
                cost_dynamics = lambda_val * cvx.norm(nu[:, k], 1)
            else:
                cost_dynamics = 0
            
            # C. State Constraint Slack (nu_s)
            cost_collision = 0
            if "nu_s" in self.variables:
                # Assuming nu_s is flattened (len_planets * K)
                # We access the slice corresponding to time step k
                n_planets = len(self.planets)
                # Striding: idx = planet_i * K + k
                # We sum the slack for all planets at this specific time k
                # indices: k, k+K, k+2K ...
                indices = [i * K + k for i in range(n_planets)]
                cost_collision += lambda_val * cvx.sum(self.variables["nu_s"][indices])

            if "nu_s_asteroids" in self.variables:
                n_asteroids = len(self.asteroids)
                indices = [i * K + k for i in range(n_asteroids)]
                cost_collision += lambda_val * cvx.sum(self.variables["nu_s_asteroids"][indices])

            # Add weighted term to integral
            integral_sum += weight * (cost_fuel + cost_dynamics + cost_collision)

        # Apply dt
        J_running = integral_sum * dt_norm

        return cvx.Minimize(J_terminal + penalty_boundary + J_running)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH                                                                                                                                                │
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)                                          │

        # The `calculate_discretization` call performs both linearization and discretization in one step.
        # It evaluates the symbolic Jacobians (from SatelliteDyn) at each point along the current trajectory guess (X_bar, U_bar)                            │
        # and then integrates the resulting continuous-time linear system to produce the discrete-time matrices (A_bar, B_plus_bar, etc.).                   │
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        # HINT: be aware that the matrices returned by calculate_discretization are flattened in F order (this way affect your code later when you use them)

        # All of these parameters need to be populated that the solver can check the constraints!

        # Populate init_state / goal_state parameter with the fixed problem boundaries
        # self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        # self.problem_parameters["goal_state"].value = self.X_bar[:, -1]
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
        self.problem_parameters["init_state"].value = x_init
        self.problem_parameters["goal_state"].value = x_goal

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
        self.problem_parameters["p_bar"].value = self.p_bar

        # You can comment out the line below to disable the verbose consistency check
        self._debug_check_flow_map_consistency(self.X_bar, self.U_bar, self.p_bar)

        # Update planet constraint parameters for the current linearization.
        num_planets = len(self.planets)
        if num_planets > 0:
            # --- Configuration Space Expansion ---
            # To ensure the entire satellite body avoids the planets, we "grow" the
            # planet's radius by the satellite's enclosing radius. We calculate this
            # from the corner of the satellite's bounding box using the user-provided formula.
            satellite_radius = np.sqrt((self.sg.w_half + self.sg.w_panel) ** 2 + (self.sg.l_f**2)) + 0.5

            # These parameters are flattened, where 'idx' = planet_idx * K + time_step_k.
            planet_C_val = np.zeros((num_planets * self.params.K, 2))
            planet_r_prime_val = np.zeros(num_planets * self.params.K)

            idx = 0
            for i, planet in enumerate(self.planets.values()):
                p_center = np.array(planet.center)
                # Use the effective radius for the constraint calculation.
                effective_radius = planet.radius + satellite_radius

                for k in range(self.params.K):
                    x_bar_k = self.X_bar[:2, k]  # (x, y) components of the reference state

                    # Calculate C_k = -2 * (x_bar_k - p_center)
                    C_k = -2 * (x_bar_k - p_center)
                    planet_C_val[idx, :] = C_k

                    # Calculate r'_k = r_eff^2 - ||x_bar_k - p_center||^2 + 2 * (x_bar_k - p_center)^T * x_bar_k
                    r_prime_k = (
                        effective_radius**2 - np.sum((x_bar_k - p_center) ** 2) + 2 * (x_bar_k - p_center) @ x_bar_k
                    )
                    planet_r_prime_val[idx] = r_prime_k

                    idx += 1

            self.problem_parameters["planet_C"].value = planet_C_val
            self.problem_parameters["planet_r_prime"].value = planet_r_prime_val

        # Update asteroid constraint parameters for the current linearization.
        num_asteroids = len(self.asteroids)
        if num_asteroids > 0:
            # Satellite radius is already calculated above for planets --> only if planets exist!
            satellite_radius = np.sqrt((self.sg.w_half + self.sg.w_panel) ** 2 + (self.sg.l_f**2))

            asteroid_C_val = np.zeros((num_asteroids * self.params.K, 2))
            asteroid_r_prime_val = np.zeros(num_asteroids * self.params.K)

            idx = 0
            for i, asteroid in enumerate(self.asteroids.values()):
                # Use the effective radius for the constraint calculation.
                effective_radius = asteroid.radius + satellite_radius

                for k in range(self.params.K):
                    # Calculate time t_k for the current time step k
                    t_k = (k / (self.params.K - 1)) * self.p_bar[0] if self.params.K > 1 else 0.0

                    # Predict asteroid position at time t_k
                    p_asteroid_center_k = self._get_asteroid_position(asteroid, t_k)

                    x_bar_k = self.X_bar[:2, k]  # (x, y) components of the reference state

                    # Calculate C_k = -2 * (x_bar_k - p_asteroid_center_k)
                    C_k = -2 * (x_bar_k - p_asteroid_center_k)
                    asteroid_C_val[idx, :] = C_k

                    # Calculate r'_k = r_eff^2 - ||x_bar_k - p_asteroid_center_k||^2 + 2 * (x_bar_k - p_asteroid_center_k)^T * x_bar_k
                    r_prime_k = (
                        effective_radius**2
                        - np.sum((x_bar_k - p_asteroid_center_k) ** 2)
                        + 2 * (x_bar_k - p_asteroid_center_k) @ x_bar_k
                    )
                    asteroid_r_prime_val[idx] = r_prime_k

                    idx += 1

            self.problem_parameters["asteroid_C"].value = asteroid_C_val
            self.problem_parameters["asteroid_r_prime"].value = asteroid_r_prime_val

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
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
        """
        """
        nu_norm = np.linalg.norm(self.variables["nu"].value)
        # nu_tc_norm = np.linalg.norm(self.variables["nu_tc"].value)
        # nu_s_k_norm = np.linalg.norm(self.variables["nu_s_k"].value)
        # total_slack = nu_norm + nu_tc_norm + nu_s_k_norm
        total_slack = nu_norm

        converged = total_slack < self.params.stop_crit
        print(f"  Total slack: {total_slack:.6e} (threshold: {self.params.stop_crit:.6e})")

        # return converged
        """

        # Criterion 1: Trajectory change
        X_star = self.variables["X"].value
        p_star = self.variables["p"].value

        # Calculate change in p
        p_change = np.linalg.norm(p_star - self.p_bar)  # L2 norm for scalar p

        # Calculate max_k(||xk - xk_bar||)
        max_x_change = 0.0
        for k in range(self.params.K):
            current_x_change = np.linalg.norm(X_star[:, k] - self.X_bar[:, k])  # L2 norm for state vector at time k
            max_x_change = max(max_x_change, current_x_change)

        trajectory_change = p_change + max_x_change
        converged_by_trajectory = trajectory_change < self.params.stop_crit

        print(f"  Trajectory change: {trajectory_change:.6e} (threshold: {self.params.stop_crit:.6e})")

        # Criterion 2: Relative predicted cost improvement
        merit_old = self._calculate_nonlinear_merit(self.X_bar, self.U_bar, self.p_bar)
        L_new = self.problem.value

        predicted_improvement = merit_old - L_new

        # Avoid division by zero or negative merit
        if merit_old > 1e-6:
            relative_improvement = predicted_improvement / merit_old
            converged_by_cost = relative_improvement < self.params.relative_stop_crit
            print(
                f"  Relative predicted improvement: {relative_improvement.item():.6e} (threshold: {self.params.relative_stop_crit:.6e})"
            )
        else:
            converged_by_cost = False
            print(f"  Relative predicted improvement: N/A (merit_old is too small)")

        return converged_by_trajectory or converged_by_cost

    def _calculate_nonlinear_merit(self, X: NDArray, U: NDArray, p: NDArray) -> float:
        """
        Calculates the nonlinear merit function Jλ(x,u,p) using a sequential defect calculation
        as described in the SCvx paper (Fig. 15).  ---  𝛿_k = x_{k+1} - 𝜓(⋅)  ---
        """
        """
        # J(cost): Time and fuel objective
        cost = (self.params.weight_p @ p).item() + self.params.weight_u * np.sum(U**2)

        # Simulate the nonlinear dynamics piecewise, starting from each X[:,k]
        # This gives us X_nl_piecewise[:, k+1] = phi(X[:, k], U[:, k], U[:, k+1], p)
        X_nl_piecewise = self.integrator.integrate_nonlinear_piecewise(X, U, p)

        # Defect calculation
        # The defect at each step is the difference between the solver's proposed next state
        # and the actual next state if we integrate one step from the solver's current state.
        # Sum the L1 norm of these defects.
        total_defect = np.sum(np.abs(X[:, 1:] - X_nl_piecewise[:, 1:]))

        # Total merit Jλ
        merit = cost + self.params.lambda_nu * total_defect
        return merit
        """
        K = self.params.K
        # Normalized time step dt for the interval [0, 1]
        dt_norm = 1.0 / (K - 1)
        lambda_val = self.params.lambda_nu

        # --- 1. Augmented Terminal Cost (Phi_lambda) ---
        # Original Terminal Cost (Time)
        J_val = (self.params.weight_p @ p).item()
        
        # Boundary Violations (Eq 50): Penalized with lambda
        init_error = np.linalg.norm(X[:, 0] - self.problem_parameters['init_state'].value, 1)
        goal_error = np.linalg.norm(X[:, -1] - self.problem_parameters['goal_state'].value, 1)
        J_val += lambda_val * (init_error + goal_error)

        # --- 2. Augmented Running Cost (Gamma_lambda) ---
        # We calculate the running terms for every step k, then apply weights.
        
        # A. Dynamics Defects (delta_k)
        # Integrate nonlinear dynamics to find actual next states
        X_nl_piecewise = self.integrator.integrate_nonlinear_piecewise(X, U, p)
        
        # Calculate defect: ||x_{k+1} - integrated(x_k)||_1
        # This results in a sequence of length K-1.
        # As per paper (text below Eq 51), nu_N is undefined, so we assume defect_N = 0.
        defects = np.sum(np.abs(X[:, 1:] - X_nl_piecewise[:, 1:]), axis=0) # Shape (K-1,)
        defects = np.append(defects, 0.0) # Shape (K,) to match time grid

        # B. State Constraints (Collisions) ([s]^+ from Eq 61)
        # We calculate the sum of violations for all planets/asteroids at each step k
        collision_violations = np.zeros(K)
        
        # Satellite radius helper
        sat_rad = np.sqrt((self.sg.w_half + self.sg.w_panel) ** 2 + (self.sg.l_f**2)) + 0.5

        # Check Planets
        if len(self.planets) > 0:
            for planet in self.planets.values():
                p_center = np.array(planet.center)
                eff_rad = planet.radius + sat_rad
                dists = np.linalg.norm(X[:2, :] - p_center.reshape(-1,1), axis=0)
                # Constraint: dist >= rad  -->  Violation: max(0, rad - dist)
                # FIX: Use squared distance to match solver's linearized constraint units
                collision_violations += np.maximum(0.0, eff_rad**2 - dists**2)

        # Check Asteroids (Time dependent)
        if len(self.asteroids) > 0:
            sat_rad_ast = np.sqrt((self.sg.w_half + self.sg.w_panel) ** 2 + (self.sg.l_f**2))
            for asteroid in self.asteroids.values():
                eff_rad = asteroid.radius + sat_rad_ast
                for k in range(K):
                    t_real = p[0] * (k / (K - 1)) if K > 1 else 0.0
                    a_pos = self._get_asteroid_position(asteroid, t_real)
                    dist = np.linalg.norm(X[:2, k] - a_pos)
                    # FIX: Use squared distance here too
                    collision_violations[k] += max(0.0, eff_rad**2 - dist**2)

        # C. Fuel Cost (Original Running Cost)
        fuel_costs = self.params.weight_u * np.sum(U**2, axis=0) # Shape (K,)

        # --- 3. Apply Trapezoidal Integration (Eq 54) ---
        # Integral = sum( weight_k * value_k ) * dt
        running_total = 0.0
        for k in range(K):
            # Eq 50: Gamma = Cost + lambda*P(defect) + lambda*P(collision)
            gamma_k = fuel_costs[k] + lambda_val * defects[k] + lambda_val * collision_violations[k]
            
            # Eq 54: Weights are 0.5 for endpoints, 1.0 for interior
            weight = 0.5 if (k == 0 or k == K - 1) else 1.0
            
            running_total += weight * gamma_k

        J_val += running_total * dt_norm

        return J_val

    def _update_trust_region(self) -> tuple[bool, dict]:
        """
        Update trust region radius and decide whether to accept or reject the step.
        Returns True if the step is accepted, False otherwise.
        Also returns a dictionary with metrics for logging.
        """
        X_star = self.variables["X"].value
        U_star = self.variables["U"].value
        p_star = self.variables["p"].value

        # Jλ(star): Nonlinear merit of the new/optimal trajectory from the solver
        merit_new = self._calculate_nonlinear_merit(X_star, U_star, p_star)

        # Jλ(bar): Nonlinear merit of the old/reference trajectory
        merit_old = self._calculate_nonlinear_merit(self.X_bar, self.U_bar, self.p_bar)

        # Lλ(star): Linearized merit from the solver's objective value
        L_new = self.problem.value

        # Predicted improvement = Jλ(bar) - Lλ(star)
        predicted_improvement = merit_old - L_new

        # Actual improvement = Jλ(bar) - Jλ(star)
        actual_improvement = merit_old - merit_new

        # You can comment out the line below to disable the trust region update print
        # self._debug_print_trust_region_update(merit_old, merit_new, L_new, actual_improvement, predicted_improvement)

        # Avoid division by zero; if predicted improvement is not positive, the step is bad.
        if predicted_improvement <= 1e-9:  # Use a small tolerance
            rho = -1.0  # Indicates a bad step
        else:
            rho = (actual_improvement / predicted_improvement).item() # Ensure float

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
        else:  # rho >= self.params.rho_2
            # Very accurate, expand trust region and accept step
            self.params.tr_radius *= self.params.beta
            accept_step = True

        # Clamp the trust region radius to its min/max values
        self.params.tr_radius = np.clip(self.params.tr_radius, self.params.min_tr_radius, self.params.max_tr_radius)

        metrics = {
            "J_bar": merit_old.item() if hasattr(merit_old, "item") else merit_old,
            "J_star": merit_new.item() if hasattr(merit_new, "item") else merit_new,
            "L_star": L_new,
            "pred_improv": predicted_improvement.item() if hasattr(predicted_improvement, "item") else predicted_improvement,
            "act_improv": actual_improvement.item() if hasattr(actual_improvement, "item") else actual_improvement,
            "rho": rho
        }

        return accept_step, metrics

    def _get_asteroid_position(self, asteroid: AsteroidParams, time: float) -> np.ndarray:
        """
        Predicts the center of an asteroid at a given time in the global frame.
        """
        orientation = asteroid.orientation
        velocity_local = np.array(asteroid.velocity)

        # Create rotation matrix to transform from local to global frame
        c, s = np.cos(orientation), np.sin(orientation)
        rot_matrix = np.array([[c, -s], [s, c]])

        # Rotate the velocity vector
        velocity_global = rot_matrix @ velocity_local

        # Predict position: start + v_global * t
        return np.array(asteroid.start) + velocity_global * time

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

    def _debug_print_defect_comparison(self):
        """
        Helper method to print a detailed comparison of defects and virtual controls.
        """
        X_bar, U_bar, p_bar = self.X_bar, self.U_bar, self.p_bar
        X_star, U_star, p_star = self.variables["X"].value, self.variables["U"].value, self.variables["p"].value
        nu_val = self.variables["nu"].value
        K = self.params.K

        # --- Defect of X_bar ---
        # Method 1: Using the nonlinear integrator `ψ`.
        X_nl_bar = self.integrator.integrate_nonlinear_piecewise(X_bar, U_bar, p_bar)
        defect_X_bar_nonlinear = X_bar[:, 1:] - X_nl_bar[:, 1:]

        # Method 2: Using the linearization from _convexification (A_bar, r_bar, etc.).
        defect_X_bar_linear = np.zeros((self.satellite.n_x, K - 1))
        for k in range(K - 1):
            A_k = cvx.reshape(
                self.problem_parameters["A_bar"][:, k], (self.satellite.n_x, self.satellite.n_x), order="F"
            ).value
            Bp_k = cvx.reshape(
                self.problem_parameters["B_plus_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
            ).value
            Bm_k = cvx.reshape(
                self.problem_parameters["B_minus_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
            ).value
            F_k = cvx.reshape(
                self.problem_parameters["F_bar"][:, k], (self.satellite.n_x, self.satellite.n_p), order="F"
            ).value
            r_k = self.problem_parameters["r_bar"][:, k].value
            x_kp1_linearized = A_k @ X_bar[:, k] + Bp_k @ U_bar[:, k + 1] + Bm_k @ U_bar[:, k] + F_k @ p_bar + r_k
            defect_X_bar_linear[:, k] = X_bar[:, k + 1] - x_kp1_linearized

        # --- Defect of X_star ---
        X_nl_star = self.integrator.integrate_nonlinear_piecewise(X_star, U_star, p_star)
        defect_X_star_nonlinear = X_star[:, 1:] - X_nl_star[:, 1:]

        print("\n[DEBUG] Defect and Virtual Control Comparison:")
        print("  -- For Reference Trajectory (X_bar) --")
        print(f"  ||defect(X_bar)|| [via nonlinear integrator]: {np.linalg.norm(defect_X_bar_nonlinear):.4e}")
        print(f"  ||defect(X_bar)|| [via linear approx]:       {np.linalg.norm(defect_X_bar_linear):.4e}")
        print(f"  --> Difference: {np.linalg.norm(defect_X_bar_nonlinear - defect_X_bar_linear):.4e}")
        print("\n  -- For Candidate Trajectory (X_star) --")
        print(f"  ||nu(X_star)||      [Virtual Control]:      {np.linalg.norm(nu_val):.4e}")
        print(f"  ||defect(X_star)||    [True Nonlinear Error]: {np.linalg.norm(defect_X_star_nonlinear):.4e}")

    def _debug_print_iteration_summary(self, i: int):
        """
        Helper method to print a summary of the current solver iteration.
        """
        if self.problem.status == "optimal":
            print(f"\n=== Iteration {i} ===")
            print(f"Total objective: {self.problem.value:.6e}")

            # Break down which slack is the problem
            nu = self.variables["nu"].value
            print(f"Slack breakdown:")
            print(f"  nu (dynamics): {np.linalg.norm(nu):.6e}, max: {np.max(np.abs(nu)):.6e}")

            # Check if we're at the boundaries
            X_new = self.variables["X"].value
            p_new = self.variables["p"].value
            print(f"Solution stats:")
            print(f"  p (time): {p_new[0]:.4f}")
            print(
                f"  U min/max: [{np.min(self.variables['U'].value):.4f}, {np.max(self.variables['U'].value):.4f}] (limits: [0, {self.sp.F_limits[1]}])"
            )
            print(f"  X[:,0] error: {np.linalg.norm(X_new[:,0] - self.problem_parameters['init_state'].value):.6e}")
            print(f"  X[:,-1] error: {np.linalg.norm(X_new[:,-1] - self.problem_parameters['goal_state'].value):.6e}")
            print(f"  Trust region: {self.params.tr_radius:.4f}")

    def _debug_print_trust_region_update(self, merit_old, merit_new, L_new, actual_improvement, predicted_improvement):
        """
        Helper method to print the details of the trust region update calculation.
        """
        # Use .item() to extract the scalar value from 1-element numpy arrays
        print("Trust region update:")
        print(f"  merit_old: {merit_old.item():.6e}, merit_new: {merit_new.item():.6e}, L_new: {L_new:.6e}")
        print(
            f"  actual_improvement: {actual_improvement.item():.6e}, predicted_improvement: {predicted_improvement.item():.6e}"
        )

    def _debug_check_flow_map_consistency(self, X_bar: NDArray, U_bar: NDArray, p_bar: NDArray):
        """
        Helper method to check if the linearization is consistent with the nonlinear dynamics
        at the reference trajectory X_bar.
        """
        K = self.params.K
        # Nonlinear one-step propagation from X_bar
        X_nl = self.integrator.integrate_nonlinear_piecewise(X_bar, U_bar, p_bar)

        print("\n[DEBUG] Flow-map consistency check (X_lin vs X_nl):")
        max_err = 0.0
        for k in range(K - 1):
            # Recover linearized matrices (unflatten)
            A_k = cvx.reshape(
                self.problem_parameters["A_bar"][:, k], (self.satellite.n_x, self.satellite.n_x), order="F"
            ).value
            Bp_k = cvx.reshape(
                self.problem_parameters["B_plus_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
            ).value
            Bm_k = cvx.reshape(
                self.problem_parameters["B_minus_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
            ).value
            F_k = cvx.reshape(
                self.problem_parameters["F_bar"][:, k], (self.satellite.n_x, self.satellite.n_p), order="F"
            ).value
            r_k = self.problem_parameters["r_bar"][:, k].value

            # Discrete linear model prediction
            X_lin_kp1 = A_k @ X_bar[:, k] + Bp_k @ U_bar[:, k + 1] + Bm_k @ U_bar[:, k] + F_k @ p_bar + r_k
            # Nonlinear flow-map propagated state
            X_nl_kp1 = X_nl[:, k + 1]
            # Difference between linear prediction and nonlinear propagation
            err = np.max(np.abs(X_lin_kp1 - X_nl_kp1))
            max_err = max(max_err, err)

        # Report the worst mismatch
        print(f"  Max linearization consistency error: {max_err:.6e}")
        if max_err > 1e-6:
            print("  WARNING: Flow map is NOT consistent with discretization!")
