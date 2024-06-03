import numpy as np

import gymnasium as gym

from wrappers.follower import FollowerWrapper


class SingleAgentLeaderWrapper(gym.Env):
    def __init__(
        self,
        env: FollowerWrapper,
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

    def _get_next_follower_action(self):
        if self.follower_epsilon_greedy and np.random.rand() < self.epsilon:
            return self.env.action_space("follower").sample()
        else:
            follower_action, _ = self.follower_model.predict(
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
class SingleAgentQueryLeaderWrapper(SingleAgentLeaderWrapper):
    def __init__(
        self,
        env: FollowerWrapper,
        queries,
        follower_model,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
    ):
        super.__init__(
            self, env, queries, follower_model, follower_epsilon_greedy, epsilon
        )

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


# Wraps a multi-agent environment to a single-agent environment from the leader's perspective
# DOEST NOT prepend the initial segment to the leader's trajectory
# The queries are assumed to be fixed
class LeaderWrapperNoInitialSegment(SingleAgentLeaderWrapper):
    def __init__(
        self,
        env: FollowerWrapper,
        queries,
        follower_model,
        leader_model=None,
        follower_epsilon_greedy: bool = False,
        epsilon: float = 0.1,
        random_follower_policy_prob: float = 0.0,
    ):
        super.__init__(
            self,
            env,
            queries,
            follower_model,
            leader_model,
            follower_epsilon_greedy,
            epsilon,
        )
        self.random_follower_policy_prob = random_follower_policy_prob

    def set_leader_model(self, leader_model):
        self.leader_model = leader_model

    def _get_next_follower_action(self):
        if self.follower_policy is not None:
            return self.follower_policy[self.last_follower_obs[0]]

        if self.follower_epsilon_greedy and np.random.rand() < self.epsilon:
            return self.env.action_space("follower").sample()
        else:
            follower_action, _ = self.follower_model.predict(
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
