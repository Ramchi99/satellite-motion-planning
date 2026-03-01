from dataclasses import dataclass
from math import sqrt
from typing import List, Set, Tuple

import numpy as np
from dg_commons import iterate_with_dt, seq_integrate, DgSampledSequence, PlayerName
from dg_commons.sim.models.spaceship import SpaceshipState
from dg_commons.sim.simulator import SimContext
from shapely.geometry import Point, Polygon

from pdm4ar.exercises_def import PerformanceResults
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget


@dataclass(frozen=True)
class PlayerMetrics(PerformanceResults):
    player_name: PlayerName
    "Player's name"
    goal_reached: bool
    """Has the player reached the goal?"""
    collided: bool
    """Has the player collided?"""
    travelled_distance: float
    """The length of the trajectory travelled by the robot"""
    episode_duration: float
    """The time it took till the end the simulation."""
    distance2goal: float
    """The beeline distance left to the goal, >=0"""
    actuation_effort: float
    """Integral of the commands sent to the robot normalized by the time taken"""
    avg_computation_time: float
    """Average computation time of the get_commands method."""


@dataclass(frozen=True)
class AvgPlayerMetrics(PerformanceResults):
    goal_success_rate: float
    """Average over the players for goal reaching rate."""
    collision_rate: float
    """Average over the players for collision with an obstacle."""
    avg_distance_travelled: float
    """The length of the trajectory travelled by the robot"""
    avg_episode_duration: float
    """The time it took till the end the simulation."""
    avg_distance2goal: float
    """The beeline distance to the goal, >=0"""
    avg_actuation_effort: float
    """Integral of the commands sent to the robot normalized by the time taken."""
    avg_computation_time: float
    """Average computation time of the get_commands method."""

    def __repr__(self):
        repr: str = ""
        for k, v in self.__dict__.items():
            value = f"{v:>5.2f}" if isinstance(v, float) else f"{v:>5}"
            repr += f"\t{k:<20}=\t" + value + ",\n"

        return f"EpisodeOutcome(\n" + repr + "\n)"

    def reduce_to_score(self) -> float:
        """Higher is better"""
        score = (self.goal_success_rate - self.collision_rate) * 1e3
        score -= self.avg_computation_time * 1e2
        score -= (
            self.avg_distance2goal / 2
            + self.avg_distance_travelled / 5
            + self.avg_episode_duration / 3
            + self.avg_actuation_effort
        ) * 1e1
        return score

        # # Weights
        # w_success = 1e3
        # w_collision = -1e3
        # w_comp_time = -1e2
        # w_dist_goal = -1e1 / 2
        # w_dist_travelled = -1e1 / 5
        # w_duration = -1e1 / 3
        # w_actuation = -1e1

        # # Contributions
        # c_success = self.goal_success_rate * w_success
        # c_collision = self.collision_rate * w_collision
        # c_comp_time = self.avg_computation_time * w_comp_time
        # c_dist_goal = self.avg_distance2goal * w_dist_goal
        # c_dist_travelled = self.avg_distance_travelled * w_dist_travelled
        # c_duration = self.avg_episode_duration * w_duration
        # c_actuation = self.avg_actuation_effort * w_actuation
        
        # score = c_success + c_collision + c_comp_time + c_dist_goal + c_dist_travelled + c_duration + c_actuation
        
        # print("\n--- Score Breakdown ---")
        # print(f"{'Metric':<25} | {'Val':<10} | {'W':<10} | {'Score':<10}")
        # print("-" * 65)
        # print(f"{'Goal Success':<25} | {self.goal_success_rate:<10.4f} | {w_success:<10.1f} | {c_success:<10.2f}")
        # print(f"{'Collision Rate':<25} | {self.collision_rate:<10.4f} | {w_collision:<10.1f} | {c_collision:<10.2f}")
        # print(f"{'Comp Time':<25} | {self.avg_computation_time:<10.4f} | {w_comp_time:<10.1f} | {c_comp_time:<10.2f}")
        # print(f"{'Dist to Goal':<25} | {self.avg_distance2goal:<10.4f} | {w_dist_goal:<10.2f} | {c_dist_goal:<10.2f}")
        # print(f"{'Dist Travelled':<25} | {self.avg_distance_travelled:<10.4f} | {w_dist_travelled:<10.2f} | {c_dist_travelled:<10.2f}")
        # print(f"{'Duration':<25} | {self.avg_episode_duration:<10.4f} | {w_duration:<10.2f} | {c_duration:<10.2f}")
        # print(f"{'Actuation':<25} | {self.avg_actuation_effort:<10.4f} | {w_actuation:<10.2f} | {c_actuation:<10.2f}")
        # print("-" * 65)
        # print(f"{'TOTAL SCORE':<25} | {'':<10} | {'':<10} | {score:<10.2f}")
        # print("-----------------------\n")

        # return score


def ex13_metrics(sim_context: SimContext) -> Tuple[AvgPlayerMetrics, List[PlayerMetrics]]:
    agents_perf: List[PlayerMetrics] = []
    collided_players: Set[PlayerName] = set()
    for cr in sim_context.collision_reports:
        collided_players.update((cr.players.keys()))

    for player_name, agent_log in sim_context.log.items():
        if "PDM4AR" not in player_name:
            continue

        states: DgSampledSequence[SpaceshipState] = agent_log.states

        # if the last state of the sim is inside the goal
        last_state = states.values[-1]
        has_reached_the_goal: bool = sim_context.missions[player_name].is_fulfilled(
            last_state, at=states.timestamps[-1]
        )

        # collision
        has_collided = True if player_name in collided_players else False

        # distance travelled
        dist: float = 0
        for it in iterate_with_dt(states):
            dist += sqrt((it.v1.x - it.v0.x) ** 2 + (it.v1.y - it.v0.y) ** 2)

        # time duration
        duration = float(states.get_end() - states.get_start())

        # distance left to goal
        last_point = Point(last_state.x, last_state.y)
        goal = sim_context.missions[player_name]

        if isinstance(goal, SpaceshipTarget):
            goal_poly: Point = Point(goal.target.x, goal.target.y)
        else:
            raise RuntimeError

        distance2goal = goal_poly.distance(last_point)

        # actuation effort
        abs_acc = agent_log.commands.transform_values(lambda u: (abs(u.F_right) + abs(u.F_left)))
        actuation_effort = seq_integrate(abs_acc).values[-1] / duration

        # computation time
        avg_comp_time = np.average(agent_log.info.values)

        # create the player metrics
        pm = PlayerMetrics(
            player_name=player_name,
            goal_reached=has_reached_the_goal,
            collided=has_collided,
            travelled_distance=dist,
            episode_duration=duration,
            distance2goal=distance2goal,
            actuation_effort=actuation_effort,
            avg_computation_time=avg_comp_time,
        )
        agents_perf.append(pm)

    goal_success_rate = [p.goal_reached for p in agents_perf].count(True) / len(agents_perf)
    collision_rate = [p.collided for p in agents_perf].count(True) / len(agents_perf)
    avg_travelled_distance = np.average([p.travelled_distance for p in agents_perf])
    avg_duration = np.average([p.episode_duration for p in agents_perf])
    avg_distance2goal = np.average([p.distance2goal for p in agents_perf])
    avg_actuation_effort = np.average([p.actuation_effort for p in agents_perf])
    avg_computation_time = np.average([p.avg_computation_time for p in agents_perf])

    avg_player_metrics = AvgPlayerMetrics(
        goal_success_rate=goal_success_rate,
        collision_rate=collision_rate,
        avg_distance_travelled=avg_travelled_distance,
        avg_episode_duration=avg_duration,
        avg_distance2goal=avg_distance2goal,
        avg_actuation_effort=avg_actuation_effort,
        avg_computation_time=avg_computation_time,
    )

    return avg_player_metrics, agents_perf
