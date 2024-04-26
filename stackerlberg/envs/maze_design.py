from __future__ import annotations

import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces

from pettingzoo import ParallelEnv


class MazeDesign(ParallelEnv):

    metadata = {
        "name": "maze_design",
    }

    def __init__(self, env: MiniGridEnv, episode_length: int = 1) -> None:
        super().__init__()

        self.agents = ["leader", "follower"]
        agent_view_area = env.agent_view_size * env.agent_view_size
        grid_area = env.width * env.height
        self.action_spaces = {
            "leader": spaces.Discrete(2**agent_view_area),
            "follower": spaces.Discrete(4),
        }
        self.observation_spaces = {
            "leader": spaces.Discrete(2**grid_area * grid_area),
            "follower": spaces.Discrete(2**agent_view_area * (agent_view_area + 1)),
        }
        self.episode_length = episode_length
        self.current_step = 0
        self.follower_position = np.array([0,0], dtype=int)
        self.goal_position = np.array([10,10], dtype=int)
        self.leader_traversed_map = np.full((env.width, env.height), False)

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def reset(self):
        self.current_step = 0
        return {"leader": 0, "follower": 0}
    
    def step(self, actions):
        self.current_step += 1

        reward_follower = -1

        done = False
        if self.follower_position == self.goal_position:
            done = True

        leader_reward = self.compute_shortest_path(self.follower_position, self.goal_position)

        rewards = {
            "leader": leader_reward,
            "follower": reward_follower
        }

        

    def compute_shortest_path(curr, goal):
        pass
        