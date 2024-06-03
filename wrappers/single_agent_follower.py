import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO

from wrappers.follower import FollowerWrapper
from utils.drone_game_obs import *


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
        leader_policy = leader_response or self._get_leader_policy()
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

    def _preprocess_observation(self, obs):
        if isinstance(self.leader_obs_space, spaces.Discrete):
            return obs
        elif isinstance(self.leader_obs_space, spaces.MultiBinary):
            return decimal_to_binary(obs, width=self.leader_obs_space.n)

    def update_leader_model(self, leader_model: PPO):
        self.leader_model = leader_model

    def _get_leader_policy(self):
        return [
            self.leader_model.predict(
                self._preprocess_observation(o), deterministic=False
            )[0]
            for o in range(self.leader_obs_size)
        ]
