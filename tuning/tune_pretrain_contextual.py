from stable_baselines3.common.evaluation import evaluate_policy
from ray import tune

from utils.config_util import load_config
from training.ppo.pretrain import pretrain_contextual, build_follower_env_contextual

config = load_config("ppo")

def objective(pretrain_config):
    follower_env = build_follower_env_contextual(config)
    follower_model = pretrain_contextual(config, pretrain_config, follower_env)

    mean, std = evaluate_policy(
        follower_model,
        follower_env,
        n_eval_episodes=1000,
    )

    return {"score": mean, "std": std}


def tune_hyperparameters():
    search_space = {
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "gamma": tune.choice([0.99, 0.95, 0.9]),
        "ent_coef": tune.choice([0.0, 0.01, 0.1, 0.5]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "n_steps": tune.choice([512, 1024, 2048, 4096]),
    }

    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=20, max_concurrent_trials=4),
    )
    results = tuner.fit()

    # save results
    df = results.get_dataframe()
    df.to_csv("tune/pretrain_ppo_results.csv")

    print(results.get_best_result(metric="score", mode="max").config)


if __name__ == "__main__":
    tune_hyperparameters()
