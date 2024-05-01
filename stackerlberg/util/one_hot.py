import numpy as np


def discrete_to_one_hot(obs, n) -> np.ndarray:
    return np.eye(n, dtype=np.int64)[obs]


def one_hot_to_discrete(one_hot) -> int:
    return np.where(one_hot == 1)[0][0]
