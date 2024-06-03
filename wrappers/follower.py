import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium import spaces
import numpy as np


# Wrapper that appends the leader's deterministic policy to the follower's observation
# Only works for small leader observation spaces
class FollowerWrapper(BaseParallelWrapper):
    def __init__(self, env, num_queries: int, leader_response: list | None = None):
        assert num_queries > 0, "num_queries must be greater than 0"
        assert leader_response is None or num_queries == len(
            leader_response
        ), "num_queries must be equal to the length of leader_response"

        super().__init__(env)
        self.num_queries = num_queries
        self.leader_response = leader_response

    def set_leader_response(self, leader_response: list):
        assert (
            len(leader_response) == self.num_queries
        ), "leader_response must be equal to the number of queries"
        self.leader_response = leader_response

    def observation_space(self, agent: str) -> spaces.Space:
        if agent == "leader":
            return self.env.observation_space(agent)

        leader_context_dims = [
            self.env.action_space("leader").n for _ in range(self.num_queries)
        ]

        if isinstance(self.env.observation_space(agent), spaces.Discrete):
            original_dims = [self.env.observation_space(agent).n]
        elif isinstance(self.env.observation_space(agent), spaces.MultiDiscrete):
            original_dims = self.env.observation_space(agent).nvec
        elif isinstance(self.env.observation_space(agent), spaces.MultiBinary):
            original_dims = [2] * self.env.observation_space(agent).n

        return spaces.MultiDiscrete([*original_dims, *leader_context_dims])

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs["follower"], np.ndarray):
            obs["follower"] = np.concatenate(
                (obs["follower"], np.array(self.leader_response))
            )
        else:
            obs["follower"] = np.array([obs["follower"], *self.leader_response])
        return obs

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if isinstance(obs["follower"], np.ndarray):
            obs["follower"] = np.concatenate(
                (obs["follower"], np.array(self.leader_response))
            )
        else:
            obs["follower"] = np.array([obs["follower"], *self.leader_response])
        return obs, rewards, terminated, truncated, infos