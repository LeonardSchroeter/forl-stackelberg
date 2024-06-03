import numpy as np
import matplotlib.pyplot as plt

from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import FollowerWrapper
from algos.tabular_q import TabularQ


class Train:
    def __init__(self, env, config=None):
        self.env = env

        def hashify_follower(obs):
            return tuple(obs)

        follower_config = config["follower"] if config else None
        leader_config = config["leader"] if config else None

        self.follower_q = TabularQ(
            env.action_space("follower").n,
            hashify=hashify_follower,
            config=follower_config,
        )
        self.leader_q = TabularQ(env.action_space("leader").n, config=leader_config)

        self.init_hyperparameters()

    def init_hyperparameters(self):
        self.leader_n_episodes = 1000
        self.follower_n_episodes = 10000

    def train_follower(self):
        for episode in range(self.follower_n_episodes):
            random_leader_policy = [
                self.env.action_space("leader").sample()
                for _ in range(self.env.observation_space("leader").n)
            ]
            self.env.set_leader_response(random_leader_policy)

            obs = self.env.reset()

            while True:
                leader_action = random_leader_policy[obs["leader"]]
                follower_action = self.env.action_space("follower").sample()

                actions = {"leader": leader_action, "follower": follower_action}

                new_obs, rewards, terminated, truncated, _ = self.env.step(actions)

                self.follower_q.update_q(
                    obs["follower"],
                    follower_action,
                    rewards["follower"],
                    new_obs["follower"],
                )

                if all(terminated.values()) or all(truncated.values()):
                    break

                obs = new_obs

    def train_leader(self):
        returns_per_episode = {"leader": [], "follower": []}

        for episode in range(self.leader_n_episodes):
            returns = self.evaluate()
            returns_per_episode["leader"].append(returns["leader"])
            returns_per_episode["follower"].append(returns["follower"])

            leader_policy = [
                self.leader_q.get_action(q, type="epsilon-greedy") for q in range(5)
            ]
            self.env.set_leader_response(leader_policy)

            leader_memory = []
            for s, a in zip(list(range(5)), leader_policy):
                leader_memory.append((s, a, 0))

            obs = self.env.reset()

            while True:
                leader_action = leader_policy[obs["leader"]]
                follower_action = self.follower_q.get_action(obs["follower"])

                actions = {"leader": leader_action, "follower": follower_action}

                new_obs, rewards, terminated, truncated, _ = self.env.step(actions)

                leader_memory.append((obs["leader"], leader_action, rewards["leader"]))

                if all(terminated.values()) or all(truncated.values()):
                    break

                obs = new_obs

            leader_quadruples = [
                (state, action, reward, next_state)
                for (state, action, reward), (next_state, _, _) in zip(
                    leader_memory, leader_memory[1:]
                )
            ]

            self.leader_q.update_q_batch(leader_quadruples)

        return returns_per_episode

    # Play one episode of the game using deterministic leader and follower policies
    def evaluate(self):
        leader_policy = [self.leader_q.get_action(q, type="greedy") for q in range(5)]

        self.env.set_leader_response(leader_policy)
        obs = self.env.reset()

        returns = {"leader": 0, "follower": 0}

        while True:
            leader_action = leader_policy[obs["leader"]]
            follower_action = self.follower_q.get_action(obs["follower"], type="greedy")

            actions = {"leader": leader_action, "follower": follower_action}

            obs, rewards, terminated, truncated, _ = self.env.step(actions)

            returns["leader"] += rewards["leader"] - self.env.reward_offset
            returns["follower"] += rewards["follower"] - self.env.reward_offset

            if all(terminated.values()) or all(truncated.values()):
                break

        return returns


good_config = {
    "follower": {
        "gamma": 0.8684315545335111,
        "alpha": 0.03649832336047994,
        "epsilon": 0.14626571202090557,
        "temperature": 1,
    },
    "leader": {
        "gamma": 0.8298757842998368,
        "alpha": 0.01903609628539893,
        "epsilon": 0.038605496690605694,
        "temperature": 1,
    },
}


if __name__ == "__main__":
    import time

    np.random.seed(int(time.time()))

    env = FollowerWrapper(
        IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
        num_queries=5,
    )
    q = Train(env, config=good_config)
    q.train_follower()
    print("Pretraining done")
    returns = q.train_leader()
    print("Training done")
    q.follower_q.print_policy()
    print(np.mean(returns["leader"]))

    plt.plot(returns["leader"], label="Leader")
    # plt.plot(returns["follower"], label="Follower")
    plt.legend()
    plt.show()
