import os

import wandb

from envs.matrix_game import IteratedMatrixGame

from envs.drone_game import DroneGame, DroneGameEnv
from wrappers.single_agent_follower import *

from wrappers.follower import ContextualPolicyWrapper

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config_args_overwrite


def build_follower_env(config, inner_outer=False):
    if config.env.name == "matrix_game":
        follower_env = ContextualPolicyWrapper(
            IteratedMatrixGame(
                matrix="prisoners_dilemma",
                episode_length=config.matrix_game.episode_len,
                memory=config.matrix_game.memory,
            ),
            num_queries=5,
        )
    elif config.env.name == "drone_game":
        env = DroneGameEnv(
            width=config.drone_game.width, height=config.drone_game.height
        )
        env = DroneGame(
            env=env,
            headless=config.drone_game.headless,
            leader_cont=config.drone_game.leader_cont,
        )
        if isinstance(env.observation_space("leader"), spaces.MultiBinary):
            num_queries = 2 ** env.observation_space("leader").n
        elif isinstance(env.observation_space("leader"), spaces.MultiDiscrete):
            num_queries = np.prod(env.observation_space("leader").nvec)
        follower_env = ContextualPolicyWrapper(
            env=env, num_queries=num_queries
        )
    if inner_outer:
        follower_env = FollowerWrapperInfoSample(follower_env)
    else:
        follower_env = SingleAgentFollowerWrapper(follower_env)

    return follower_env


def pretrain(config, pretrain_config, follower_env=None):
    if follower_env is None:
        follower_env = build_follower_env(config)

    if config.training.log_wandb:
        run = wandb.init(project="stackelberg-ppo-follower", sync_tensorboard=True)

    if config.drone_game.leader_cont:
        folder = "leader_cont"
    else:
        folder = ""

    follower_model, callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, folder, "follower"),
        follower_env,
        config.training.log_wandb,
        pretrain_config,
        run.id,
    )

    follower_model.learn(
        total_timesteps=300_00,
        reset_num_timesteps=False,
        callback=callback_list,
    )

    return follower_model


if __name__ == "__main__":
    config = config = load_config_args_overwrite("configs/ppo.yml")

    pretrain_config = {
        "learning_rate": lambda progress: config.training.pretrain_start_lr
        * (1 - progress)
        + config.training.pretrain_end_lr * progress,
        "gamma": config.training.pretrain_gamma,
        "ent_coef": config.training.pretrain_ent_coef,
        "batch_size": config.training.pretrain_batch_size,
        "n_steps": config.training.pretrain_n_steps,
    }

    pretrain(config, pretrain_config)
