import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.utils.wrappers import BaseParallelWrapper
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.matrix_game import IteratedMatrixGame


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

        if isinstance(self.env.observation_space(agent), spaces.Discrete):
            return spaces.MultiDiscrete(
                [
                    self.env.observation_space(agent).n,
                    *[
                        self.env.action_space("leader").n
                        for _ in range(self.num_queries)
                    ],
                ]
            )
        elif isinstance(self.env.observation_space(agent), spaces.MultiDiscrete):
            return spaces.MultiDiscrete(
                [
                    *self.env.observation_space(agent).nvec,
                    *[
                        self.env.action_space("leader").n
                        for _ in range(self.num_queries)
                    ],
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


# Wrapper that adds the last action and reward to the follower's observation and plays multiple episodes
# Optionally zeros out the leaders reward in all episodes except the last one
# This allows Meta-RL using a recurrent policy
class FollowerWrapperMetaRL(BaseParallelWrapper):
    def __init__(
        self,
        env,
        num_episodes: int,
        max_reward: float = np.inf,
        min_reward: float = -np.inf,
        zero_leader_reward: bool = True,
    ):
        assert num_episodes >= 2, "num_episodes must be greater than or equal to 2"

        super().__init__(env)
        self.env = env
        self.num_episodes = num_episodes
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.zero_leader_reward = zero_leader_reward

        self.current_episode = 1
        self.reset_next = False

    def observation_space(self, agent: str) -> spaces.Space:
        if agent == "leader":
            return self.env.observation_space(agent)

        return gym.spaces.MultiBinary(n=self.env.observation_space(agent).n + 2)

    def _inner_reset(self):
        self.current_episode += 1
        self.reset_next = False
        obs = self.env.reset()
        obs["follower"] = np.concatenate((obs["follower"], [0, 0]))

        return obs

    def reset(self):
        self.current_episode = 0
        return self._inner_reset()

    def step(self, actions):
        if self.reset_next:
            obs = self._inner_reset()
            reward = {"follower": 0, "leader": 0}
            term = {"follower": False, "leader": False}
            trunc = {"follower": False, "leader": False}
            info = {"follower": {}, "leader": {}}
        else:
            obs, reward, term, trunc, info = self.env.step(actions)
            obs["follower"] = np.concatenate(
                (obs["follower"], [actions["follower"], actions["leader"]])
            )

        if self.zero_leader_reward and self.current_episode < self.num_episodes:
            reward["leader"] = 0

        terminated = False
        if any(term.values()) or any(trunc.values()):
            self.reset_next = True
            if self.current_episode == self.num_episodes:
                terminated = True
        term = {"follower": terminated, "leader": terminated}
        trunc = {"follower": False, "leader": False}

        return obs, reward, term, trunc, info

    def render():
        pass

    def close():
        pass


if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapperMetaRL(env, num_episodes=2)
    obs = env.reset()

    i = 1

    while True:
        i += 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        new_obs, rewards, terminated, truncated, infos = env.step(actions)

        print("STEP")
        print(obs)
        print(actions)
        print(rewards)
        print(new_obs)

        if any(terminated.values()) or any(truncated.values()):
            break

        obs = new_obs
    print("Done", i)
