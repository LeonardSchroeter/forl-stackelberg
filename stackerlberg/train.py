import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces

from envs.matrix_game import IteratedMatrixGame
from envs.drone_game import DroneGame, DroneGameEnv
from wrappers.follower import FollowerWrapper
from algorithms.tabular_q import TabularQ

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--env", help="choose you environment: matgame, dronegame")
parser.add_argument("--headless", help="disable GUI", action="store_true")
parser.add_argument("--pretrain", action="store_true")
# parser.add_argument("--resume_pretrain", action="store_true")
# parser.add_argument("--test_pretrain", action="store_true")
parser.add_argument("--train", action="store_true")
# parser.add_argument("--resume_train", action="store_true")
parser.add_argument("--test_train", action="store_true")
args = parser.parse_args()

import pickle

class Train:
    def __init__(self, env: FollowerWrapper, config=None):
        self.env = env
        self.num_queries = env.num_queries

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
        self.leader_n_episodes = 10000
        self.follower_n_episodes = 10000

    def leader_obs_to_int(self, obs):
        if isinstance(self.env.observation_space("leader"), spaces.Discrete):
            obs_leader = obs
        elif isinstance(self.env.observation_space("leader"), spaces.MultiBinary):
            binary_str = "".join(str(int(bit)) for bit in obs)[::-1]
            obs_leader = int(binary_str, base=2)

        return obs_leader

    def train_follower(self):
        if isinstance(self.env.observation_space("leader"), spaces.Discrete):
            n = self.env.observation_space("leader").n
        elif isinstance(self.env.observation_space("leader"), spaces.MultiBinary):
            n = 2**self.env.observation_space("leader").n
        for episode in range(self.follower_n_episodes):
            print(f"Episode: {episode}")
            random_leader_policy = [
                self.env.action_space("leader").sample()
                for _ in range(n)
            ]
            self.env.set_leader_response(random_leader_policy)

            obs = self.env.reset()

            while True:
                obs_leader = self.leader_obs_to_int(obs["leader"])
                leader_action = random_leader_policy[obs_leader]
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

        with open(f"checkpoints_q/follower_q_{args.env}.pt", "wb") as file:
            pickle.dump(self.follower_q.q_table, file)

    def train_leader(self):
        with open(f"checkpoints_q/follower_q_{args.env}.pt", "rb") as file:
            self.follower_q.q_table = pickle.load(file)
        returns_per_episode = {"leader": [], "follower": []}

        for episode in range(self.leader_n_episodes):
            print(f"Episode: {episode}")
            returns = self.evaluate()
            returns_per_episode["leader"].append(returns["leader"])
            returns_per_episode["follower"].append(returns["follower"])

            leader_policy = [
                self.leader_q.get_action(q, type="epsilon-greedy") for q in range(self.num_queries)
            ]
            self.env.set_leader_response(leader_policy)

            leader_memory = []
            for s, a in zip(list(range(self.num_queries)), leader_policy):
                if args.env == "matgame":
                    leader_memory.append(s, a, 0)
                elif args.env == "dronegame":
                    s_binary_str = np.binary_repr(s, width=self.env.observation_space("leader").n)
                    s_arr = np.array([float(bit) for bit in s_binary_str])
                    leader_memory.append((s_arr, a, 0))

            obs = self.env.reset()

            while True:
                obs_leader = self.leader_obs_to_int(obs["leader"])
                leader_action = leader_policy[obs_leader]
                follower_action = self.follower_q.get_action(obs["follower"])

                actions = {"leader": leader_action, "follower": follower_action}

                new_obs, rewards, terminated, truncated, _ = self.env.step(actions)

                leader_memory.append((obs["leader"], leader_action, rewards["leader"]))

                if all(terminated.values()) or all(truncated.values()):
                    break

                obs = new_obs

            leader_quadruples = [
                (tuple(state), action, reward, tuple(next_state))
                for (state, action, reward), (next_state, _, _) in zip(
                    leader_memory, leader_memory[1:]
                )
            ]

            self.leader_q.update_q_batch(leader_quadruples)

        with open(f"checkpoints_q/leader_q_{args.env}.pt", "wb") as file:
            pickle.dump(self.leader_q.q_table, file)

        return returns_per_episode

    # Play one episode of the game using deterministic leader and follower policies
    def evaluate(self):
        
        leader_policy = [self.leader_q.get_action(q, type="greedy") for q in range(self.env.num_queries)]
        self.env.set_leader_response(leader_policy)
        obs = self.env.reset()

        returns = {"leader": 0, "follower": 0}

        while True:
            obs_leader = self.leader_obs_to_int(obs["leader"])
            leader_action = leader_policy[obs_leader]
            follower_action = self.follower_q.get_action(obs["follower"], type="greedy")

            actions = {"leader": leader_action, "follower": follower_action}

            obs, rewards, terminated, truncated, _ = self.env.step(actions)

            if hasattr(self.env, "reward_offset"):
                returns["leader"] += rewards["leader"] - self.env.reward_offset
                returns["follower"] += rewards["follower"] - self.env.reward_offset

            if all(terminated.values()) or all(truncated.values()):
                break

        return returns

    def test_policy_on_env(self):

        with open(f"checkpoints_q/follower_q_{args.env}.pt", "rb") as file:
            self.follower_q.q_table = pickle.load(file)
        with open(f"checkpoints_q/leader_q_{args.env}.pt", "rb") as file:
            self.leader_q.q_table = pickle.load(file)

        leader_policy = [self.leader_q.get_action(q, type="greedy") for q in range(self.env.num_queries)]
        self.env.set_leader_response(leader_policy)

        self.env.env.headless = False
        obs = self.env.reset()
        while True:
            leader_action = self.leader_q.get_action(tuple(obs["leader"]))
            follower_action = self.follower_q.get_action(tuple(obs["follower"]))
            actions = {"leader": leader_action, "follower": follower_action}
            new_obs, rewards, terminated, truncated, _ = self.env.step(actions)
            print(obs, actions, rewards)
            obs = new_obs
            if all(terminated.values()) or all(truncated.values()):
                break

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
    if args.env == "matgame":
        env = FollowerWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
        )
    elif args.env == "dronegame":
        grid_size = 20
        max_steps = 2 * grid_size
        env = DroneGameEnv(size=grid_size, max_steps=max_steps, agent_start_pos=(5,10))
        env = DroneGame(env=env, headless=args.headless)
        env = FollowerWrapper(env=env, num_queries=2**env.observation_space("leader").n)
    q = Train(env, config=good_config)
    if args.pretrain:
        q.train_follower()
        print("Pretraining done")
    if args.train:
        returns = q.train_leader()
        print("Training done")
        print(np.mean(returns["leader"]))
        plt.plot(returns["leader"], label="Leader")
        # plt.plot(returns["follower"], label="Follower")
        plt.legend()
        plt.show()
    if args.test_train:
        if args.env == "matgame":
            q.follower_q.print_policy_matgame()
        elif args.env == "dronegame":
            q.test_policy_on_env()


