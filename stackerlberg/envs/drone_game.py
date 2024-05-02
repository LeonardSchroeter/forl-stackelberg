from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple

import time

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces

from pettingzoo import ParallelEnv

from util.point import Point2D


class DroneGameEnv(MiniGridEnv):
    def __init__(
        self,
        size=20,
        agent_start_pos=(1, 10),
        agent_start_dir=0,
        agent_dir_fixed=True,
        agent_view_size=3,
        max_steps: int | None = None,
        drone_options: List = [(13, 4), (13, 8), (13, 12), (13, 16)],
        num_divisions=4,
        drone_cover_size=3,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_dir_fixed = agent_dir_fixed
        self.agent_view_size = agent_view_size
        self.drone_options = drone_options
        self.num_divisions = num_divisions
        self.drone_cover_size = drone_cover_size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "drone game"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        for j in range(1, self.width - 1):
            self.put_obj(Goal(), self.height - 2, j)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "drone game"

    def in_grid(self, point: Point2D, exclude_goal_line=True):
        return point >= Point2D(1, 1) and point <= Point2D(
            self.width - 2 - int(exclude_goal_line),
            self.height - 2,
        )


class DroneGame(ParallelEnv):
    metadata = {
        "name": "drone_game",
    }

    def __init__(self, env: DroneGameEnv, headless: bool = False) -> None:
        super().__init__()

        self.env = env
        self.headless = headless
        if not headless:
            self.env.render_mode = "human"

        self.agents = ["leader", "follower"]

        agent_view_area = env.agent_view_size * env.agent_view_size
        # grid_area = (env.width - 2) * (env.height - 2) # The outer are walls so -2

        # leader action: which of prescribed places to place drone
        # follower action: fwd(0), bwd(1), left(2), right(3)
        self.action_spaces = {
            "leader": spaces.Discrete(len(self.env.drone_options)),
            "follower": spaces.Discrete(4),
        }
        # leader observation: which division does the follower lie in
        # follower observation: wall occupancy in its local view size
        self.observation_spaces = {
            "leader": spaces.MultiBinary(self.env.num_divisions),
            "follower": spaces.MultiDiscrete([4] * agent_view_area),
        }

        self.drones = []

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def reset(self):
        self.env.reset()
        return {"leader": np.zeros(self.env.num_divisions), 
                "follower": np.zeros(self.env.agent_view_size * self.env.agent_view_size)}

    def step(self, actions):
        if not self.headless:
            print(f"\n STEP: {self.env.step_count}\n")

        observations, rewards = {}, {}

        if len(self.drones) >= 4:
            dead_drone = self.drones.pop(0)
            dead_drone.undo_lava()
            del dead_drone

        # if not self.headless:
        #     self.env.render()
        #     time.sleep(0.5)

        if not self.headless:
            print("Leader takes action")
        self.leader_act(actions["leader"])
        observations["leader"] = self.get_leader_observation()
        rewards["leader"] = self.get_leader_reward()

        if not self.headless:
            print(f"leader observation: {observations['leader']}")
            print(f"leader reward: {rewards['leader']}")

        for drone in self.drones:
            drone.undo_lava()
        for drone in self.drones:
            drone.brownian_motion()
        for drone in self.drones:
            drone.set_lava()

        if not self.headless:
            self.env.render()
            time.sleep(0.5)
            print("\nFollower takes action")

        self.follower_act(actions["follower"])
        if not self.headless:
            self.env.render()
            time.sleep(0.5)
        observations["follower"] = self.get_follower_observation()
        rewards["follower"] = self.get_follower_reward()

        if not self.headless:
            print(f"follower observation: {observations['follower']}")
            print(f"follower observation binary: {observations['follower']}")
            print(f"follower reward: {rewards['follower']}")

        terminated = (
            (self.env.step_count >= self.env.max_steps)
            or (self.env.agent_pos[0] == self.env.height - 2)
            or isinstance(
                self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Lava
            )
        )

        if not self.headless:
            print(f"\nterminated: {terminated}")

        terminated = {"leader": terminated, "follower": terminated}
        truncated = {"leader": False, "follower": False}
        info = {"leader": {}, "follower": {}}

        return observations, rewards, terminated, truncated, info

    def leader_act(self, action):
        drone_place = self.env.drone_options[action]
        self.drones.append(Drone(env=self.env, radius=1, center=drone_place))

    # def binary_matrix_to_int(self, binary_matrix: np.array):
    #     binary_vec = binary_matrix.T.flatten()
    #     binary_str = "".join(str(bit) for bit in binary_vec)[::-1]
    #     int_number = int(binary_str, base=2)
    #     return int_number

    def get_leader_observation(self):
        observation = np.zeros(self.env.num_divisions)

        for drone in self.drones:
            dists = [
                drone.center.euclidean_distance(Point2D(i, j))
                for i, j in self.env.drone_options
            ]
            index = np.argmin(dists)
            observation[index] = 1

        return observation

    def get_leader_reward(self):
        reward = self.env.height - 2 - self.env.agent_pos[0]

        if isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Lava
        ):
            reward += self.env.height

        # TODO: Do we even need this? The reward already scales by distance to the goal
        if isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Goal
        ):
            reward -= self.env.height

        return reward

    def follower_act(self, action):
        self.env.render_mode = None
        match action:
            case 0:  # fwd
                self.env.step(self.env.actions.forward)
            case 1:  # bwd
                self.env.step(self.env.actions.left)
                self.env.step(self.env.actions.left)
                self.env.step(self.env.actions.forward)
                if self.env.agent_dir_fixed:
                    self.env.step(self.env.actions.right)
                    self.env.step(self.env.actions.right)
            case 2:  # left
                self.env.step(self.env.actions.left)
                self.env.step(self.env.actions.forward)
                if self.env.agent_dir_fixed:
                    self.env.step(self.env.actions.right)
            case 3:  # right
                self.env.step(self.env.actions.right)
                self.env.step(self.env.actions.forward)
                if self.env.agent_dir_fixed:
                    self.env.step(self.env.actions.left)
        self.env.render_mode = "human" if not self.headless else None
        if not self.headless:
            self.env.render()

    def get_follower_observation(self):
        topX, topY, botX, botY = self.env.get_view_exts()

        observation = np.zeros((self.env.agent_view_size, self.env.agent_view_size))
        for i in range(topX, botX):
            for j in range(topY, botY):
                i_local, j_local = self.env.relative_coords(i, j)

                if not self.env.in_grid(Point2D(i, j)):
                    observation[i_local, j_local] = 1
                elif isinstance(self.env.grid.get(i, j), Lava):
                    observation[i_local, j_local] = 2
                elif isinstance(self.env.grid.get(i, j), Wall):
                    observation[i_local, j_local] = 3

        return observation.flatten()

    def get_follower_reward(self):
        reward = self.env.agent_pos[0]

        if isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Lava
        ):
            reward -= self.env.height

        if isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Goal
        ):
            reward += self.env.height

        return reward


class Drone:
    def __init__(self, env: DroneGameEnv, center: Tuple[int, int], radius=1) -> None:
        self.radius = radius
        # self.to_death = lifespan
        self.center = Point2D(*center)
        self.env = env
        self.set_lava()

    def in_grid(self, point: Point2D = None):
        point = point or self.center
        return (
            self.env.in_grid(point + Point2D(self.radius, self.radius))
            and self.env.in_grid(point + Point2D(self.radius, -self.radius))
            and self.env.in_grid(point + Point2D(-self.radius, self.radius))
            and self.env.in_grid(point + Point2D(-self.radius, -self.radius))
        )

    def fill_body(self, type):
        for i in range(
            self.center.x - self.radius,
            self.center.x + self.radius + 1,
        ):
            for j in range(
                self.center.y - self.radius,
                self.center.y + self.radius + 1,
            ):
                if self.env.in_grid(Point2D(i, j)):
                    if type is None:
                        self.env.grid.set(i, j, None)
                    else:
                        self.env.put_obj(type, i, j)

    def set_lava(self):
        self.fill_body(Lava())

    def undo_lava(self):
        self.fill_body(None)

    def brownian_motion(self):
        # self.to_death -= 1

        dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        dirs = list(filter(lambda x: self.in_grid(self.center + Point2D(*x)), dirs))

        index = np.random.randint(len(dirs))
        move = Point2D(*dirs[index])
        next_center = self.center + move

        self.center = next_center


if __name__ == "__main__":
    env = DroneGameEnv(agent_start_pos=(3, 10), agent_dir_fixed=True)
    env = DroneGame(env=env)

    follower_action_seq = [0, 0, 0, 0, 0, 2, 2, 0, 0, 3, 3, 1, 1, 0, 0, 0, 0]

    i = 0
    env.env.reset()
    while True:
        actions = {}
        # actions["follower"] = env.action_space("follower").sample()
        if i < len(follower_action_seq):
            actions["follower"] = follower_action_seq[i]
        else:
            actions["follower"] = 0
        actions["leader"] = i % 4
        observation, reward, terminated, _, _ = env.step(actions)
        if terminated["follower"]:
            break
        i += 1
