import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO

class SingleAgentFollowerWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space("follower")
        self.observation_space = env.observation_space("follower")
        self.last_leader_obs = None

    def reset(self, leader_response=None, seed=None, options=None):
        if isinstance(self.env.observation_space("leader"), spaces.Discrete):
            leader_policy = leader_response or [
                self.env.action_space("leader").sample()
                for _ in range(self.env.observation_space("leader").n)
            ]
        elif isinstance(self.env.observation_space("leader"), spaces.MultiBinary):
            if leader_response is not None:
                leader_policy = leader_response
            else:
                leader_policy = [
                    self.env.action_space("leader").sample()
                    for _ in range(2 ** self.env.observation_space("leader").n)
                ]
        self.env.set_leader_response(leader_policy)
        obs = self.env.reset()
        self.last_leader_obs = obs["leader"]
        return obs["follower"], {}

    def step(self, action):
        if isinstance(self.env.observation_space("leader"), spaces.Discrete):
            last_leader_obs = self.last_leader_obs
        elif isinstance(self.env.observation_space("leader"), spaces.MultiBinary):
            binary_str = "".join(str(int(bit)) for bit in self.last_leader_obs)[::-1]
            last_leader_obs = int(binary_str, base=2)
        actions = {
            "follower": action,
            "leader": self.env.leader_response[last_leader_obs],
        }
        obs, reward, term, trunc, info = self.env.step(actions)
        self.last_leader_obs = obs["leader"]

        return (
            obs["follower"],
            reward["follower"],
            term["follower"],
            trunc["follower"],
            info["follower"],
        )

    def render():
        pass

    def close():
        pass


class SingleAgentLeaderWrapper(gym.Env):
    def __init__(self, env, queries, follower_model):
        self.env = env
        self.queries = queries
        self.follower_model = follower_model

        self.current_step = 0
        self.last_follower_obs = None
        self.leader_response = []

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.last_follower_obs = None
        self.leader_response = []
        return self.queries[0], {}

    def step(self, action):
        self.current_step += 1

        if self.current_step <= len(self.queries):
            self.leader_response.append(action)

        if self.current_step < len(self.queries):
            return self.queries[self.current_step], 0, False, False, {}
        elif self.current_step == len(self.queries):
            self.env.set_leader_response(self.leader_response)
            obs = self.env.reset()
            self.last_follower_obs = obs["follower"]
            return obs["leader"], 0, False, False, {}

        follower_action, _states = self.follower_model.predict(
            self.last_follower_obs, deterministic=True
        )
        actions = {
            "leader": action,
            "follower": follower_action,
        }
        obs, reward, term, trunc, info = self.env.step(actions)
        self.last_follower_obs = obs["follower"]

        return (
            obs["leader"],
            reward["leader"],
            term["leader"],
            trunc["leader"],
            info["leader"],
        )

    def render():
        pass

    def close():
        pass


class InfoSampleSingleAgentFollowerWrapper(SingleAgentFollowerWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.leader_model: PPO = None
    
    def update_leader_model(self, leader_model):
        self.leader_model = leader_model

    def reset(
        self, leader_response=None, seed=None, options=None
    ):

        if isinstance(self.env.observation_space("leader"), spaces.Discrete):
            leader_policy = leader_response or [
                self.leader_model.predict(o, deterministic=False)[0]
                for o in range(self.env.observation_space("leader").n)
            ]
        elif isinstance(self.env.observation_space("leader"), spaces.MultiBinary):
            if leader_response is not None:
                leader_policy = leader_response
            else:
                for o in range(2 ** self.env.observation_space("leader").n):
                    o_bin = [int(bit) for bit in np.binary_repr(o, self.env.observation_space("leader").n)][::-1]
                    leader_policy = [
                        self.leader_model.predict(o_bin, deterministic=False)[0]
                        for _ in range(2 ** self.env.observation_space("leader").n)
                    ]

        self.env.set_leader_response(leader_policy)
        obs = self.env.reset()
        self.last_leader_obs = obs["leader"]
        return obs["follower"], {}
