import gymnasium as gym
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from follower import FollowerWrapper


# Wraps a multi-agent environment to a single-agent environment from the follower's perspective
# Assumes that the environment is a FollowerWrapper, i.e. the leaders policy is fixed
# Optionally, the leader's policy is passed to the wrapped environment
# The leader's observations space has to be discrete
class SingleAgentFollowerWrapper(gym.Env):
    def __init__(
        self, env: FollowerWrapper, recursively_set_leader_response: bool = True
    ):
        self.env = env
        self.recursively_set_leader_response = recursively_set_leader_response

        self.action_space = env.action_space("follower")
        self.observation_space = env.observation_space("follower")
        self.last_leader_obs = None
        self.leader_response = None

    def set_leader_response(self, leader_response):
        self.leader_response = leader_response
        if self.recursively_set_leader_response:
            self.env.set_leader_response(leader_response)

    def reset(self, leader_response=None, seed=None, options=None):
        # leader_policy = leader_response or [
        #     self.env.action_space("leader").sample()
        #     for _ in range(self.env.observation_space("leader").n)
        # ]
        leader_policy = [0, 0, 0, 1, 1]
        self.set_leader_response(leader_policy)
        obs = self.env.reset()
        self.last_leader_obs = obs["leader"]
        return obs["follower"], {}

    def step(self, action):
        actions = {
            "follower": action,
            "leader": self.leader_response[self.last_leader_obs],
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
    def __init__(
        self,
        env,
        queries,
        follower_model,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
    ):
        self.env = env
        self.queries = queries
        self.follower_model = follower_model
        self.follower_epsilon_greedy = follower_epsilon_greedy
        self.epsilon = epsilon

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

        self.last_follower_obs = None

    def _get_next_follower_action(self):
        if self.follower_epsilon_greedy and np.random.rand() < self.epsilon:
            return self.env.action_space("follower").sample()
        else:
            follower_action, _states = self.follower_model.predict(
                self.last_follower_obs, deterministic=True
            )
            return follower_action

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        self.last_follower_obs = obs["follower"]
        return obs["leader"], {}

    def step(self, action):
        follower_action = self._get_next_follower_action()
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


# Wraps a multi-agent environment to a single-agent environment from the leader's perspective
# Prepends the initial segment to the leader's trajectory
# The queries are assumed to be fixed
class SingleAgentQueryLeaderWrapper(gym.Env):
    def __init__(
        self,
        env,
        queries,
        follower_model,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
    ):
        self.env = env
        self.queries = queries
        self.follower_model = follower_model
        self.follower_epsilon_greedy = follower_epsilon_greedy
        self.epsilon = epsilon

        self.current_step = 0
        self.last_follower_obs = None
        self.leader_response = []

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

    def _get_next_follower_action(self):
        if self.follower_epsilon_greedy and np.random.rand() < self.epsilon:
            return self.env.action_space("follower").sample()
        else:
            follower_action, _states = self.follower_model.predict(
                self.last_follower_obs, deterministic=True
            )
            return follower_action

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

        follower_action = self._get_next_follower_action()
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


# Wraps a multi-agent environment to a single-agent environment from the leader's perspective
# DOEST NOT prepend the initial segment to the leader's trajectory
# The queries are assumed to be fixed
class LeaderWrapperNoInitialSegment(gym.Env):
    def __init__(
        self,
        env,
        queries,
        follower_model,
        leader_model=None,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
        random_follower_policy_prob: float = 0.0,
    ):
        self.env = env
        self.queries = queries
        self.follower_model = follower_model
        self.leader_model = leader_model
        self.follower_epsilon_greedy = follower_epsilon_greedy
        self.epsilon = epsilon
        self.random_follower_policy_prob = random_follower_policy_prob

        self.last_follower_obs = None
        self.follower_policy = None

        self.action_space = env.action_space("leader")
        self.observation_space = env.observation_space("leader")

    def set_leader_model(self, leader_model):
        self.leader_model = leader_model

    def _get_next_follower_action(self):
        if self.follower_policy is not None:
            return self.follower_policy[self.last_follower_obs[0]]

        if self.follower_epsilon_greedy and np.random.rand() < self.epsilon:
            return self.env.action_space("follower").sample()
        else:
            follower_action, _states = self.follower_model.predict(
                self.last_follower_obs, deterministic=True
            )
            return follower_action

    def reset(self, seed=None, options=None):
        if (
            self.random_follower_policy_prob > 0
            and np.random.rand() < self.random_follower_policy_prob
        ):
            self.follower_policy = [
                self.env.action_space("follower").sample() for _ in range(5)
            ]
        else:
            self.follower_policy = None

        leader_response = [
            self.leader_model.predict(query)[0] for query in self.queries
        ]
        self.env.set_leader_response(leader_response)
        obs = self.env.reset()
        self.last_follower_obs = obs["follower"]
        return obs["leader"], {}

    def step(self, action):
        follower_action = self._get_next_follower_action()
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
