import numpy as np
from ray import tune

from training.train_tabular_q import TrainTabQ
from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import ContextualPolicyWrapper


def objective(config):
    scores = []
    for i in range(10):
        env = ContextualPolicyWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
        )
        q = TrainTabQ(env, config=config)
        q.train_follower()
        returns = q.train_leader()
        score = np.mean(returns["leader"])
        scores.append(score)
    score = np.mean(scores)
    return {"score": score}


search_space = {
    "follower": {
        "gamma": tune.uniform(0.8, 1),
        "alpha": tune.uniform(0, 0.2),
        "epsilon": tune.uniform(0, 0.2),
        "temperature": 1,
    },
    "leader": {
        "gamma": tune.uniform(0.8, 1),
        "alpha": tune.uniform(0, 0.2),
        "epsilon": tune.uniform(0, 0.2),
        "temperature": 1,
    },
}


def tune_hyperparameters():
    tuner = tune.Tuner(
        objective, param_space=search_space, tune_config=tune.TuneConfig(num_samples=10)
    )
    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="max").config)


if __name__ == "__main__":
    tune_hyperparameters()
