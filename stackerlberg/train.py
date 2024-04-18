from typing import Union, Literal
import numpy as np

from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import FollowerWrapper


class TabularQ:
    def __init__(self, env):
        self.env = env
        self.follower_q_table = {}

        for i in range(5):
            for j in range(32):
                j_bits = (int(x) for x in list(np.binary_repr(j, width=5)))
                self.follower_q_table[(i, tuple(j_bits))] = np.zeros(self.env.action_space("follower").n)

        self.leader_q_table = {}

        for i in range(5):
            self.leader_q_table[i] = np.zeros(self.env.action_space("leader").n)

        self.init_hyperparameters()

    def init_hyperparameters(self):
        self.n_episodes = 10000
        self.temperature = 1.0
        self.gamma = 0.95
        self.alpha = 0.1

    def hashify_follower(self, obs):
        return (obs["follower"]["original"], tuple(obs["follower"]["queries"]))

    def get_follower_action(self, obs, type: Literal["greedy", "softmax"] = "greedy"):
        obs_key = self.hashify_follower(obs)
        q_values = self.follower_q_table[obs_key]

        if type == "greedy":
            return np.argmax(q_values)
        
        q_values = np.exp(q_values / self.temperature)
        q_values = q_values / np.sum(q_values)
        return np.random.choice(self.env.action_space("follower").n, p=q_values)

    def get_leader_action(self, obs, type: Literal["greedy", "softmax"] = "greedy"):
        obs_key = obs["leader"]
        q_values = self.leader_q_table[obs_key]

        if type == "greedy":
            return np.argmax(q_values)
        
        q_values = np.exp(q_values / self.temperature)
        q_values = q_values / np.sum(q_values)
        return np.random.choice(self.env.action_space("leader").n, p=q_values)

    def pretrain(self):
        for episode in range(self.n_episodes):
            random_deterministic_leader_policy = np.array([self.env.action_space("leader").sample() for _ in range(self.env.observation_space("leader").n)])
            self.env.set_leader_response(random_deterministic_leader_policy)

            obs = self.env.reset()

            while True:
                leader_action = random_deterministic_leader_policy[obs["leader"]]
                follower_action = self.env.action_space("follower").sample()

                actions = {
                    "leader": leader_action,
                    "follower": follower_action
                }

                new_obs, rewards, terminated, truncated, _ = self.env.step(actions)

                obs_key = self.hashify_follower(obs)
                new_obs_key = self.hashify_follower(new_obs)
                
                self.follower_q_table[obs_key][follower_action] = (1 - self.alpha) * self.follower_q_table[obs_key][follower_action] + self.alpha * (rewards["follower"] + self.gamma * np.max(self.follower_q_table[new_obs_key]))

                if all(terminated.values()) or all(truncated.values()):
                    break

                obs = new_obs

    def train(self):
        queries = [0, 1, 2, 3, 4]
        for episode in range(self.n_episodes * 10):
            memory = []

            query_answers = [self.get_leader_action({"leader": q}, type="softmax") for q in queries]
            if episode % 1000 == 0:
                print(f"Episode {episode}, query answers: {query_answers}")

            self.env.set_leader_response(query_answers)

            for s, a in zip(queries, query_answers):
                memory.append((
                    {"leader": s},
                    {"leader": a},
                    {"leader": 0}
                ))

            obs = self.env.reset()

            while True:
                leader_action = query_answers[obs["leader"]]
                follower_action = self.get_follower_action(obs)

                actions = {
                    "leader": leader_action,
                    "follower": follower_action
                }

                new_obs, rewards, terminated, truncated, _ = self.env.step(actions)

                memory.append((obs, actions, rewards))

                if all(terminated.values()) or all(truncated.values()):
                    break

                obs = new_obs

            for i in range(len(memory) - 1):
                obs, actions, rewards = memory[i]
                next_obs, _, _ = memory[i + 1]

                self.leader_q_table[obs["leader"]][actions["leader"]] = (1 - self.alpha) * self.leader_q_table[obs["leader"]][actions["leader"]] + self.alpha * (rewards["leader"] + self.gamma * np.max(self.leader_q_table[next_obs["leader"]]))

if __name__ == "__main__":
    env = FollowerWrapper(IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2), num_queries=5)
    q = TabularQ(env)
    q.pretrain()
    print("Pretraining done")
    """
    for j in range(32):
        for i in range(5):
            j_bits = tuple(int(x) for x in list(np.binary_repr(j, width=5)))
            print(f"State: {i}, {j_bits}")
            print("Action values:", q.follower_q_table[(i, j_bits)])
            print("Chosen action:", q.get_follower_action({"follower": {"original": i, "queries": list(j_bits)}}))
    """
    q.train()
    print("Training done")


    for i in range(5):
        print(f"State: {i}")
        print("Action values:", q.leader_q_table[i])
        print("Chosen action:", q.get_leader_action({"leader": i}))