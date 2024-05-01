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
        agent_view_size=3,
        max_steps: int | None = None,
        drone_options: List = [(13, 4), (13, 8), (13, 12), (13, 16)],
        num_divisions=4,
        drone_cover_size=3,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
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
        # follower action: fwd(0), left(1), right(2)
        self.action_spaces = {
            "leader": spaces.Discrete(len(self.env.drone_options)),
            "follower": spaces.Discrete(4),
        }
        # leader observation: which division does the follower lie in
        # follower observation: wall occupancy in its local view size
        self.observation_spaces = {
            "leader": spaces.Discrete(2**self.env.num_divisions),
            "follower": spaces.Discrete(2**agent_view_area),
        }

        self.drones = []

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def reset(self):
        self.env.reset()
        return {"leader": 0, "follower": 0}

    def step(self, actions):
        print(f"\n STEP: {self.env.step_count}\n")

        observations, rewards = {}, {}

        if len(self.drones) >= 4:
            dead_drone = self.drones.pop(0)
            dead_drone.undo_lava()
            del dead_drone

        # if not self.headless:
        #     self.env.render()
        #     time.sleep(0.5)

        print("Leader takes action")
        self.leader_act(actions["leader"])
        observations["leader"] = self.get_leader_observation()
        rewards["leader"] = self.get_leader_reward()

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

        print("Follower takes action")
        self.follower_act(actions["follower"])
        if not self.headless:
            self.env.render()
            time.sleep(0.5)
        observations["follower"] = self.get_follower_observation()
        rewards["follower"] = -rewards["leader"] + self.env.height

        print(f"follower observation: {observations['follower']}")
        print(
            f"follower observation binary: {np.binary_repr(observations['follower'])}"
        )
        print(f"follower reward: {rewards['follower']}")

        terminated = (
            (self.env.step_count >= self.env.max_steps)
            or (self.env.agent_pos[0] == self.env.height - 2)
            or isinstance(
                self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Lava
            )
        )

        print(f"terminated: {terminated}")

        return observations, rewards, terminated, {}, {}

    def leader_act(self, action):
        drone_place = self.env.drone_options[action]
        self.drones.append(Drone(env=self.env, radius=1, center=drone_place))

    def binary_matrix_to_int(self, binary_matrix: np.array):
        binary_vec = binary_matrix.T.flatten()
        binary_str = "".join(str(bit) for bit in binary_vec)[::-1]
        int_number = int(binary_str, base=2)
        return int_number

    def get_leader_observation(self):
        m = (
            self.env.agent_pos[0] < self.env.width / 2,
            self.env.agent_pos[1] < self.env.height / 2,
        )
        if m == (1, 1):
            observation = 0
        elif m == (1, 0):
            observation = 1
        elif m == (0, 1):
            observation = 2
        elif m == (0, 0):
            observation = 3

        return observation

    def get_leader_reward(self):
        reward = self.env.height - 1 - self.env.agent_pos[0]

        if isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Lava
        ):
            reward += self.env.height

        if isinstance(
            self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Goal
        ):
            reward -= self.env.height

        return reward

    def follower_act(self, action):
        match action:
            case 0:  # fwd
                self.env.step(self.env.actions.forward)
            case 1:  # bwd
                self.env.render_mode = None
                self.env.step(self.env.actions.left)
                self.env.step(self.env.actions.left)
                self.env.render_mode = "human" if not self.headless else None
                self.env.step(self.env.actions.forward)
            case 2:  # left
                self.env.render_mode = None
                self.env.step(self.env.actions.left)
                self.env.render_mode = "human" if not self.headless else None
                self.env.step(self.env.actions.forward)
            case 3:  # right
                self.env.render_mode = None
                self.env.step(self.env.actions.right)
                self.env.render_mode = "human" if not self.headless else None
                self.env.step(self.env.actions.forward)

    def get_follower_observation(self):
        topX, topY, botX, botY = self.env.get_view_exts()

        local_lava = np.full((self.env.agent_view_size, self.env.agent_view_size), 0)
        for i in range(topX, botX):
            for j in range(topY, botY):
                i_local, j_local = self.env.relative_coords(i, j)
                if not self.env.in_grid(
                    Point2D(i, j)
                ):  # Out of grid map: deemed as lava (TODO: improve)
                    local_lava[i_local, j_local] = 1
                else:
                    tile = self.env.grid.get(i, j)
                    if isinstance(tile, Lava):
                        local_lava[i_local, j_local] = 1

        observation = self.binary_matrix_to_int(local_lava)

        return observation


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
            self.env.in_grid(self.center + Point2D(self.radius, self.radius))
            and self.env.in_grid(self.center + Point2D(self.radius, -self.radius))
            and self.env.in_grid(self.center + Point2D(-self.radius, self.radius))
            and self.env.in_grid(self.center + Point2D(-self.radius, -self.radius))
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

    # def get_follower_reward(self):
    #     pass


if __name__ == "__main__":
    env = DroneGameEnv(agent_start_pos=(3, 10))
    env = DroneGame(env=env)

    i = 0
    env.env.reset()
    while True:
        actions = {}
        actions["follower"] = env.action_space("follower").sample()
        actions["leader"] = i % 4
        observation, reward, terminated, _, _ = env.step(actions)
        if terminated:
            break
        i += 1
