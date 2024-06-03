from typing import Literal
import numpy as np

default_config = {
    "gamma": 0.99,
    "alpha": 0.05,
    "epsilon": 0.1,
    "temperature": 1.0,
}


def is_hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False


class TabularQ:
    def __init__(self, n_actions, hashify=lambda x: x, config=default_config):
        self.init_hyperparameters(config)
        self.n_actions = n_actions
        self.q_table = {}
        self.hashify = hashify

    def init_hyperparameters(self, config):
        self.gamma = config["gamma"]
        self.alpha = config["alpha"]
        self.epsilon = config["epsilon"]
        self.temperature = config["temperature"]

    def set_q(self, state, action, value):
        if not is_hashable(state):
            state = self.hashify(state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)

        self.q_table[state][action] = value

    def get_q(self, state, action=None):
        if not is_hashable(state):
            state = self.hashify(state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)

        if action is None:
            return self.q_table[state]
        else:
            return self.q_table[state][action]

    def get_action(
        self, state, type: Literal["greedy", "softmax", "epsilon-greedy"] = "greedy"
    ):
        q_values = self.get_q(state)

        if type == "greedy":
            return np.argmax(q_values)
        elif type == "epsilon-greedy":
            if np.random.rand() < self.epsilon:
                return np.random.choice(np.arange(self.n_actions))
            else:
                return np.argmax(q_values)
        else:
            q_values = np.exp(q_values / self.temperature)
            q_values = q_values / np.sum(q_values)
            return np.random.choice(np.arange(self.n_actions), p=q_values)

    def update_q(self, state, action, reward, next_state):
        q_value = self.get_q(state, action)
        next_q_value = np.max(self.get_q(next_state))

        self.set_q(
            state,
            action,
            q_value + self.alpha * (reward + self.gamma * next_q_value - q_value),
        )

    def update_q_batch(self, trajectory):
        for state, action, reward, next_state in trajectory:
            self.update_q(state, action, reward, next_state)

    def print_policy(self):
        for j in range(32):
            j_bits = [int(x) for x in list(np.binary_repr(j, width=5))][::-1]
            response = []
            for i in range(5):
                action = self.get_action([i, *j_bits], type="greedy")
                response.append(action)
            print(f"{j_bits} -> {response}")
