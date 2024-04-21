import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

import numpy as np

from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium import spaces

from envs.matrix_game import IteratedMatrixGame


class FollowerWrapper(BaseParallelWrapper):
    def __init__(self, env, num_queries: int):
        if num_queries <= 0:
            raise ValueError("num_queries must be greater than 0")

        super().__init__(env)
        self.env = env
        self.num_queries = num_queries 

    def set_leader_response(self, leader_response):
        if self.num_queries != len(leader_response):
            raise ValueError("num_queries must be equal to the length of leader_response")
        
        self.leader_response = leader_response

    def observation_space(self, agent: str) -> spaces.Space:
        if agent == "leader":
            return self.env.observation_space(agent)

        # return spaces.Dict(
        #     original=self.env.observation_space(agent),
        #     queries=spaces.Tuple(
        #         [self.env.action_space("leader") for _ in range(self.num_queries)]
        #     ),
        # )
        return spaces.Tuple((self.env.observation_space(agent)), spaces.Tuple(
                [self.env.action_space("leader") for _ in range(self.num_queries)]))
    
    def action_space(self, agent: str) -> spaces.Space:
        return self.env.action_space(agent)

    def reset(self):
        obs = self.env.reset()
        # obs["follower"] = {"original": obs["follower"], "queries": self.leader_response}
        obs_follower = self.leader_response.tolist()
        obs_follower.insert(0, obs["follower"])
        obs["follower"] = np.array(obs_follower, dtype=np.float64)
        return obs

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        # obs["follower"] = {"original": obs["follower"], "queries": self.leader_response}
        # obs["leader"] = obs["leader"].item()
        obs_follower = self.leader_response.tolist()
        obs_follower.insert(0, obs["follower"])
        obs["follower"] = np.array(obs_follower, dtype=np.float64)
        return obs, rewards, terminated, truncated, infos

if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapper(env, num_queries=3, leader_response=[0, 1, 0])
    obs = env.reset()

    while True:
        obs, rewards, terminated, truncated, infos = env.step(
            {agent: env.action_space(agent).sample() for agent in env.agents}
        )

        if any(terminated.values()) or any(truncated.values()):
            break
    print("Done")