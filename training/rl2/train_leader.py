import os

import argparse

import yaml

# import ppo form stable baselines
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from rl2.utils.setup_experiment import create_env, get_policy_net_for_inference
from rl2.utils.checkpoint_util import _latest_step
from rl2.utils.rmckp_callback import RmckpCallback

from rl2.envs.stackelberg.trial_wrapper import TrialWrapper
from rl2.envs.stackelberg.leader_env import SingleAgentLeaderWrapper

from train import add_args

import wandb
from wandb.integration.sb3 import WandbCallback


def create_argparser():
    parser = argparse.ArgumentParser(description="""Training script for RL^2.""")

    ### Environment
    parser.add_argument(
        "--environment",
        choices=["matrix_game", "drone_game"],
        default="matrix_game",
    )
    parser.add_argument(
        "--max_episode_len",
        type=int,
        default=10,
        help="Timesteps before automatic episode reset. "
        + "Ignored if environment is bandit.",
    )
    parser.add_argument(
        "--num_meta_episodes", type=int, default=3, help="Episodes per meta-episode."
    )

    ### Architecture
    parser.add_argument(
        "--architecture", choices=["gru", "lstm", "snail", "transformer"], default="gru"
    )
    parser.add_argument("--num_features", type=int, default=256)

    ### Checkpointing
    parser.add_argument("--model_name", type=str, default="defaults")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser


def train(env, config):
    run = wandb.init(project="rl2-leader", sync_tensorboard=True)

    ckp_path = "checkpoints/leader"

    if os.listdir(ckp_path):
        latest_step = _latest_step(ckp_path)
        model_name = f"checkpoints/leader/model_{latest_step}_steps.zip"
        print("Loading leader model from " + model_name)
        model = PPO.load(model_name, env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
        )

    total_timesteps = 1000_0000

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=ckp_path,
        save_replay_buffer=True,
        save_vecnormalize=True,
        name_prefix="model",
    )
    rmckp_callback = RmckpCallback(ckp_path=ckp_path)

    callback_list = [checkpoint_callback, rmckp_callback]
    if config.training.log_wandb:
        wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        callback_list.append(wandb_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callback_list),
        reset_num_timesteps=False,
    )

    model.save(f"checkpoints/leader_ppo_{config.env.name}")


if __name__ == "__main__":
    file_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_dir, "rl2", "envs", "config.yml"), "rb") as file:
        config = yaml.safe_load(file.read())

    config = add_args(config)

    follower_env = create_env(config=config)

    policy_net = get_policy_net_for_inference(follower_env, config)

    env = TrialWrapper(follower_env._env, num_episodes=3)
    env = SingleAgentLeaderWrapper(env, follower_policy_net=policy_net)

    train(env, config)
