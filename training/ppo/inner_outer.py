import os

import wandb

from wrappers.single_agent_follower import *

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config_args_overwrite

from training.ppo.pretrain import build_follower_env
from training.ppo.train_leader import build_leader_env_ppo


def maybe_load_model(config, follower_training_config, run_id=0):
    follower_env = build_follower_env(config, inner_outer=True)

    follower_model, follower_callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "inner_outer", "follower"),
        follower_env,
        config.training.log_wandb & (run_id != 0),
        follower_training_config,
        run_id,
    )

    leader_env = build_leader_env_ppo(
        config=config, follower_env=follower_env, follower_model=follower_model
    )

    leader_model, leader_callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "inner_outer", "leader"),
        leader_env,
        log_wandb=False,
        run_id=run_id,
    )

    return (
        follower_env,
        leader_env,
        follower_model,
        leader_model,
        follower_callback_list,
        leader_callback_list,
    )


def train(config, follower_training_config):
    
    if config.training.log_wandb:
        run = wandb.init(project="stackelberg-ppo-inner-outer", sync_tensorboard=True)
    
    (
        follower_env,
        leader_env,
        follower_model,
        leader_model,
        follower_callback_list,
        leader_callback_list,
    ) = maybe_load_model(config, follower_training_config, run.id)

    leader_env.set_follower_model(follower_model)
    follower_env.set_leader_model(leader_model)

    for _ in range(100):
        
        follower_model.learn(
            total_timesteps=5000,
            reset_num_timesteps=False,
            callback=follower_callback_list,
        )
        
        leader_model.learn(
            total_timesteps=1000,
            reset_num_timesteps=False,
            callback=leader_callback_list,
        )


if __name__ == "__main__":
    config = load_config_args_overwrite("configs/ppo.yml")
    follower_training_config = {
        "learning_rate": lambda progress: config.training.pretrain_start_lr
        * (1 - progress)
        + config.training.pretrain_end_lr * progress,
        "gamma": config.training.pretrain_gamma,
        "ent_coef": config.training.pretrain_ent_coef,
        "batch_size": config.training.pretrain_batch_size,
        "n_steps": config.training.pretrain_n_steps,
    }

    train(config, follower_training_config)
