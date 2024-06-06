import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO

from wrappers.follower import FollowerWrapper
from utils.drone_leader_observation import *

import os

class SingleAgentFollowerWrapper(gym.Env):
    def __init__(self, env: FollowerWrapper):
        self.env = env

        self.observation_space = self.env.observation_space("follower")
        self.action_space = self.env.action_space("follower")

        self.leader_obs_space = self.env.observation_space("leader")
        self.leader_act_space = self.env.action_space("leader")

        if isinstance(self.leader_obs_space, spaces.Discrete):
            self.leader_obs_size = self.leader_obs_space.n
        elif isinstance(self.leader_obs_space, spaces.MultiBinary):
            self.leader_obs_size = 2**self.leader_obs_space.n

    @property
    def plant(self):
        return self.env.plant

    def _get_leader_policy(self):
        return [self.leader_act_space.sample() for _ in range(self.leader_obs_size)]

    def reset(self, leader_response=None, seed=None, options=None):
        leader_policy = (
            self._get_leader_policy() if leader_response is None else leader_response
        )
        self.env.set_leader_response(leader_policy)

        obs = self.env.reset()
        self.last_leader_obs = obs["leader"]

        return obs["follower"], {}

    def _get_next_leader_action(self):
        if isinstance(self.leader_obs_space, spaces.Discrete):
            last_leader_obs = self.last_leader_obs
        elif isinstance(self.leader_obs_space, spaces.MultiBinary):
            last_leader_obs = binary_to_decimal(self.last_leader_obs)
        return self.env.leader_response[last_leader_obs]

    def step(self, action):
        actions = {
            "follower": action,
            "leader": self._get_next_leader_action(),
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


class FollowerWrapperInfoSample(SingleAgentFollowerWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.leader_obs_space, spaces.MultiBinary):  # only drone game
            obs_size = self.env.observation_space("leader").n
            obin_file_path = (
                f"envs/dronegame_obin_size{obs_size}.npy"
            )
        if os.path.isfile(obin_file_path):
            self.obs_bin = np.load(obin_file_path)
        else:
            self.obs_bin = []
            for o in range(2**obs_size):
                print(o)
                self.obs_bin.append(
                    decimal_to_binary(o, width=obs_size)
                )
            np.save(obin_file_path, self.obs_bin)

    def _preprocess_observation(self, obs):
        if isinstance(self.leader_obs_space, spaces.Discrete):
            return obs
        elif isinstance(self.leader_obs_space, spaces.MultiBinary):
            return self.obs_bin[obs]

    def set_leader_model(self, leader_model: PPO):
        self.leader_model = leader_model

    def _get_leader_policy(self):
        return [
            self.leader_model.predict(
                self._preprocess_observation(o), deterministic=False
            )[0]
            for o in range(self.leader_obs_size)
        ]
