import os

from stable_baselines3 import PPO
import numpy as np
import yaml
import argparse
import wandb

from envs.matrix_game import IteratedMatrixGame

from envs.drone_game import DroneGame, DroneGameEnv
from wrappers.single_agent_follower import *

from wrappers.follower import FollowerWrapper

from utils.checkpoint_util import maybe_load_checkpoint_ppo


def add_args(config):
    parser = argparse.ArgumentParser(description="""Pretraining script for PPO.""")

    parser.add_argument(
        "--name",
        choices=["bandit", "tabular_mdp", "matrix_game", "drone_game",
        ],
        default="bandit",
    )

    args = parser.parse_args()
    for key, value in vars(args).items():
        for key_config, value_config in config.items():
            if key in value_config.keys():
                config[key_config][key] = value

    for key in config.keys():
        config[key] = argparse.Namespace(**config[key])
    config = argparse.Namespace(**config)

    return config


with open("configs/ppo.yml", "rb") as file:
    config = yaml.safe_load(file.read())
config = add_args(config)

checkpoint_path = f"checkpoints/{config.env.name}/ppo"

if config.training.log_wandb:
    run = wandb.init(project="stackelberg-ppo", sync_tensorboard=True)


def build_follower_env():
    if config.env.name == "matrix_game":
        follower_env = FollowerWrapper(
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
        env = DroneGame(env=env, headless=config.drone_game.headless)
        follower_env = FollowerWrapper(
            env=env, num_queries=2 ** env.observation_space("leader").n
        )
    follower_env = SingleAgentFollowerWrapper(follower_env)

    return follower_env


def pretrain(follower_env, pretrain_config):
    follower_model, callback_list = maybe_load_checkpoint_ppo(
        checkpoint_path + "/follower",
        follower_env,
        config.training.log_wandb,
        pretrain_config,
        run.id,
    )

    # leader_env = build_leader_env(
    #     follower_env=follower_env, follower_model=follower_model
    # )

    # leader_model = PPO("MlpPolicy", leader_env, verbose=1)

    # for _ in range(100):
    # follower_env.update_leader_model(leader_model)
    follower_model.learn(
        total_timesteps=30000,
        reset_num_timesteps=False,
        callback=callback_list,
    )
    # leader_env.update_follower_model(follower_model)
    # leader_model.learn(total_timesteps=1000, reset_num_timesteps=False)
    # leader_model.save(os.path.join(checkpoints_path, "leader_pretrained"))
    return follower_model


# def test_pretrain(env):
#     checkpoints_path = f"checkpoints/{args.env}/follower"
#     latest_step = _latest_step(checkpoints_path)
#     model = PPO.load(
#         os.path.join(checkpoints_path, f"ppo_{latest_step}_steps.zip"),
#         env=env,
#     )
#     if args.env == "matgame":
#         for response in [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]:
#             for s in range(5):
#                 obs = [s, *response]
#                 action = model.predict(obs, deterministic=True)[0]
#                 print(f"state: {s}, context: {response}, action: {action}")
#     elif args.env == "dronegame":
#         env.env.env.headless = False
#         env.env.env.verbose = True
#         leader_response = np.full((2**4,), 1, dtype=int)
#         leader_response = np.full((2**12,), 0, dtype=int)
#         # leader_response = np.array(
#         #     [0, 3, 0, 3, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3], dtype=int
#         #     # [3, 1, 3, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 1], dtype=int
#         # )
#         obs, _ = env.reset(leader_response=leader_response)

#         while True:
#             action = model.predict(obs, deterministic=True)[0]
#             new_obs, rewards, terminated, truncated, _ = env.step(action)
#             print(obs, action, rewards)
#             obs = new_obs

#             if terminated or truncated:
#                 break

#         env.env.env.env.close(video_name="size6_context_drone0.avi")


# def build_leader_env(follower_env, follower_model=None):
#     if follower_model is None:
#         latest_step_follower = _latest_step(f"checkpoints/{args.env}/follower")
#         follower_model = PPO.load(
#             f"checkpoints/{args.env}/follower/ppo_{latest_step_follower}_steps.zip"
#         )
#     if args.env == "matgame":
#         leader_env = SingleAgentLeaderWrapper(
#             follower_env.env,
#             queries=[0, 1, 2, 3, 4],
#             follower_model=follower_model,
#             remove_initseg=args.remove_initseg,
#         )
#     else:
#         num_queries = 2 ** follower_env.env.observation_space("leader").n
#         queries = [
#             [
#                 int(bit)
#                 for bit in np.binary_repr(
#                     i, width=follower_env.env.observation_space("leader").n
#                 )
#             ][::-1]
#                 for bit in np.binary_repr(
#                     i, width=follower_env.env.observation_space("leader").n
#                 )
#             ][::-1]
#             for i in range(num_queries)
#         ]
#         leader_env = SingleAgentLeaderWrapper(
#             follower_env.env,
#             queries=queries,
#             follower_model=follower_model,
#             remove_initseg=args.remove_initseg,
#         )

#     return leader_env


# def train(leader_env):
#     leader_ckppath = f"checkpoints/{args.env}/leader"
#     if not os.path.exists(leader_ckppath):
#         os.makedirs(leader_ckppath)

#     if not os.listdir(leader_ckppath):
#         leader_model = PPO.load(
#             f"checkpoints/{args.env}/leader_pretrained.zip", env=leader_env
#         , env=leader_env)
#     else:
#         latest_step = _latest_step(leader_ckppath)
#         leader_model = PPO.load(
#             os.path.join(leader_ckppath, f"ppo_{latest_step}_steps.zip"), env=leader_env
#         )

#     if leader_model.tensorboard_log is None:
#         leader_model.tensorboard_log = f"runs/{run.id}"

#     checkpoint_callback = CheckpointCallback(
#         save_freq=1000,
#         save_path=leader_ckppath,
#         name_prefix="ppo",
#         save_replay_buffer=True,
#         save_vecnormalize=True,
#     )
#     rmckp_callback = RmckpCallback(ckp_path=leader_ckppath)

#     callback_list = [checkpoint_callback, rmckp_callback]
#     if args.log_wandb:
#         wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
#         callback_list.append(wandb_callback)

#     leader_model.learn(
#         total_timesteps=200_000,
#         callback=CallbackList(callback_list),
#     )
#     if args.remove_initseg:
#         leader_model.save(f"checkpoints/leader_ppo_woinitseg")
#     else:
#         leader_model.save(f"checkpoints/leader_ppo_{args.env}")


# def test_train(leader_env):
#     leader_ckppath = f"checkpoints/{args.env}/leader"
#     if (not os.path.exists(leader_ckppath)) or (not os.listdir(leader_ckppath)):
#         model_path = f"checkpoints/{args.env}/leader_pretrained.zip"
#     else:
#         latest_step = _latest_step(leader_ckppath)
#         model_path = os.path.join(leader_ckppath, f"ppo_{latest_step}_steps.zip")

#     leader_model = PPO.load(model_path, env=leader_env)
#     print("Loading model " + model_path + "\n")

#     if args.remove_initseg and not leader_env.leader_response:
#         leader_env.set_leader_response(leader_model)

#     if args.remove_initseg and not leader_env.leader_response:
#         leader_env.set_leader_response(leader_model)

#     if args.env != "matgame":
#         leader_env.env.env.headless = False
#         leader_env.env.env.verbose = True
#         leader_env.env.env.sleep_time = 0.7

#     # play a single episode to check learned leader and follower policies
#     obs, _ = leader_env.reset()
#     while True:
#         action = leader_model.predict(obs, deterministic=True)[0]
#         new_obs, rewards, terminated, truncated, _ = leader_env.step(action)
#         print(obs, action, rewards)
#         obs = new_obs

#         if terminated or truncated:
#             break
#     leader_env.env.env.env.close(video_name="complete_round.avi")


# def print_policy():
#     leader_model = PPO.load(f"checkpoints/leader_ppo_{args.env}")
#     for o in range(16):
#         o_bin = [int(bit) for bit in np.binary_repr(o, 4)][::-1]
#         o_bin = [int(bit) for bit in np.binary_repr(o, 4)][::-1]
#         action = leader_model.predict(o_bin, deterministic=True)[0]
#         print(f"obs: {o_bin}, act: {action}")


if __name__ == "__main__":
    follower_env = build_follower_env()

    pretrain_config = {
        "learning_rate": lambda progress: config.training.pretrain_start_lr
        * (1 - progress)
        + config.training.pretrain_end_lr * progress,
        "gamma": config.training.pretrain_gamma,
        "ent_coef": config.training.pretrain_ent_coef,
        "batch_size": config.training.pretrain_batch_size,
        "n_steps": config.training.pretrain_n_steps,
    }

    pretrain(follower_env=follower_env, pretrain_config=pretrain_config)
    # if args.test_pretrain:
    #     test_pretrain(env=follower_env)

    # leader_env = build_leader_env(follower_env=follower_env)
    # if args.train:
    #     train(leader_env=leader_env)
    # if args.test_train:
    #     test_train(leader_env=leader_env)
    #     print_policy()
    # leader_env = build_leader_env(follower_env=follower_env)
    # if args.train:
    #     train(leader_env=leader_env)
    # if args.test_train:
    #     test_train(leader_env=leader_env)
    #     print_policy()
