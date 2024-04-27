import gymnasium as gym


class SingleAgentFollowerWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space("follower")
        self.observation_space = env.observation_space("follower")
        self.last_leader_obs = None

    def reset(self, leader_response=None, seed=None, options=None):
        leader_policy = leader_response or [
            self.env.action_space("leader").sample()
            for _ in range(self.env.observation_space("leader").n)
        ]
        self.env.set_leader_response(leader_policy)
        obs = self.env.reset()
        self.last_leader_obs = obs["leader"]
        return obs["follower"], {}

    def step(self, action):
        actions = {
            "follower": action,
            "leader": self.env.leader_response[self.last_leader_obs],
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
