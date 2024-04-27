import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium import spaces

from envs.matrix_game import IteratedMatrixGame


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

        return spaces.MultiDiscrete(
            [
                self.env.observation_space(agent).n,
                *[self.env.action_space("leader").n for _ in range(self.num_queries)],
            ]
        )

    def reset(self):
        obs = self.env.reset()
        obs["follower"] = [obs["follower"], *self.leader_response]
        return obs

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        obs["follower"] = [obs["follower"], *self.leader_response]
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
