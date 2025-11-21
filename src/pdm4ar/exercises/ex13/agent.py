import numpy as np
from dataclasses import dataclass
from typing import Sequence
from decimal import Decimal

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

from pdm4ar.exercises.ex13.planner import SatellitePlanner
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
from pdm4ar.exercises_def.ex13.utils_plot import plot_traj


# HINT: as a good practice we suggest to use the config class to centralise activation of the debugging options
class Config:
    PLOT = True
    VERBOSE = False


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """
    # --- Parameters for Simple L2 Norm Replanning ---
    my_tol: float = 0.1

    # --- Parameters for Sophisticated Weighted Norm Replanning ---
    pos_threshold: float = 0.1  # meters
    vel_threshold: float = 0.2  # m/s
    angle_threshold: float = 5.0 * np.pi / 180.0  # 2 degrees in radians
    ang_vel_threshold: float = 5.0 * np.pi / 180.0 # 5 deg/s in radians


class SatelliteAgent(Agent):
    # How does it enter in the simulation? The SpaceshipAgent object is created as value
    # corresponding to key "PDM4ARSpaceship" in dict "players", which is an attribute of
    # SimContext returned by "sim_context_from_yaml" in utils_config.py
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SatelliteState
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SatelliteCommands]
    state_traj: DgSampledSequence[SatelliteState]
    myname: PlayerName
    planner: SatellitePlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: MyAgentParams # Declare params attribute
    t_replan: float # Time of the last replan

    def __init__(
        self,
        init_state: SatelliteState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SatelliteAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.planets = planets
        self.asteroids = asteroids
        self.params = MyAgentParams() # Initialize MyAgentParams here
        self.t_replan = Decimal('0.0') # The first plan starts at t=0

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.

        the time spent in this method is **not** considered in the score.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.planner = SatellitePlanner(planets=self.planets, asteroids=self.asteroids, sg=self.sg, sp=self.sp)
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        # make sure you consider both types of goals accordingly
        # (Docking is a subclass of SpaceshipTarget and may require special handling
        # to take into account the docking structure)
        self.goal_state = init_sim_obs.goal.target

        # Plot docking station (this is optional, for better visualization)
        if Config.PLOT and isinstance(init_sim_obs.goal, DockingTarget):
            A, B, C, A1, A2, half_p_angle = init_sim_obs.goal.get_landing_constraint_points()
            init_sim_obs.goal.plot_landing_points(A, B, C, A1, A2)

        #
        # TODO: Implement Compute Initial Trajectory
        #

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)

        NOTE: this function is not run in real time meaning that the simulation is stopped when the function is called.
        Thus the time efficiency of the replanning is not critical for the simulation.
        However the average time spent in get_commands is still considered in the score.

        Do **not** modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        self.actual_trajectory.append(current_state)
        # expected_state = self.state_traj.at_interp(sim_obs.time)

        # Use relative time for trajectory lookup
        relative_time = sim_obs.time - self.t_replan
        expected_state = self.state_traj.at_interp(relative_time)

        # plotting the trajectory every 2.5 sec (this is optional, for better visualization)
        if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)

        # --- CHOOSE REPLANNING STRATEGY ---

        # --- Strategy 1: Simple L2 Norm (Euclidean Distance) ---
        # UNCOMMENT THE BLOCK BELOW TO USE THIS STRATEGY
        # # 1. Calculate deviation between current and expected state
        # current_state_vec = np.array([current_state.x, current_state.y, current_state.psi, current_state.vx, current_state.vy, current_state.dpsi])
        # expected_state_vec = np.array([expected_state.x, expected_state.y, expected_state.psi, expected_state.vx, expected_state.vy, expected_state.dpsi])
        # deviation = np.linalg.norm(current_state_vec - expected_state_vec)
        #
        # # 2. Check if deviation exceeds the threshold
        # if deviation > self.params.my_tol:
        #     print(f"Replanning at time {sim_obs.time:.2f} s due to deviation of {deviation:.3f} m (threshold: {self.params.my_tol:.3f})")
        #     self.cmds_plan, self.state_traj = self.planner.compute_trajectory(current_state, self.goal_state)
        #     self.t_replan = sim_obs.time
        #     relative_time = 0.0

        # --- Strategy 2: Sophisticated Weighted Infinity Norm (Recommended) ---
        # COMMENT OUT THE BLOCK BELOW IF YOU USE STRATEGY 1
        # 1. Calculate deviation using a weighted infinity norm.
        current_state_vec = np.array([current_state.x, current_state.y, current_state.psi, current_state.vx, current_state.vy, current_state.dpsi])
        expected_state_vec = np.array([expected_state.x, expected_state.y, expected_state.psi, expected_state.vx, expected_state.vy, expected_state.dpsi])
        
        error_vec = current_state_vec - expected_state_vec
        
        # IMPORTANT: Correctly handle angle wrap-around for psi (error is in [-pi, pi])
        error_vec[2] = (error_vec[2] + np.pi) % (2 * np.pi) - np.pi

        # Define the vector of thresholds from our params
        thresholds = np.array([
            self.params.pos_threshold, self.params.pos_threshold,
            self.params.angle_threshold,
            self.params.vel_threshold, self.params.vel_threshold,
            self.params.ang_vel_threshold
        ])

        # Normalize each error component by its threshold
        normalized_errors = np.abs(error_vec) / thresholds

        # The deviation metric is the maximum of the normalized errors (this is the infinity norm)
        deviation_metric = np.max(normalized_errors)

        # 2. Check if the unitless deviation metric exceeds 1.0
        if deviation_metric > 1.0:
            print(f"Replanning at time {sim_obs.time:.2f} s. Deviation metric: {deviation_metric:.3f} > 1.0")
            # 3. Re-compute trajectory from the current state
            self.cmds_plan, self.state_traj = self.planner.compute_trajectory(current_state, self.goal_state)
            
            # Update replan time and reset relative time for this step
            self.t_replan = sim_obs.time
            relative_time = 0.0 # The new plan starts now, so we are at its t=0
        # --- End of Replanning Implementation ---

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        # cmds = self.cmds_plan.at_interp(sim_obs.time)
        # Use relative time for command lookup
        cmds = self.cmds_plan.at_interp(relative_time)

        return cmds

        # return SatelliteCommands(
        #     F_left=1, F_right=1
        # )  # can be replaced by SatelliteCommands(F_left=1, F_right=1) if you want to test constant commands
