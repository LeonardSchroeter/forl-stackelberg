import os

from stable_baselines3 import PPO
import numpy as np
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from envs.matrix_game import IteratedMatrixGame

# from envs.maze_design import Maze, MazeDesign
from envs.drone_game import DroneGame, DroneGameEnv
from wrappers.single_agent import (
    InfoSampleSingleAgentFollowerWrapper,
    SingleAgentLeaderWrapper,
)
from wrappers.follower import FollowerWrapper

from util.checkpoint import _latest_step
from util.rmckp_callback import RmckpCallback

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="choose you environment: matgame, dronegame")
parser.add_argument("--headless", help="disable GUI", action="store_true")
parser.add_argument("--verbose", help="anable outputs", action="store_true")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--test_pretrain", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--test_train", action="store_true")
parser.add_argument("--log_wandb", action="store_true")
args = parser.parse_args()

if args.log_wandb and (args.pretrain or args.train):
    run = wandb.init(project="forl-stackerlberg", sync_tensorboard=True)


def build_follower_env():
    if args.env == "matgame":
        follower_env = FollowerWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
        )
    elif args.env == "dronegame":
        env = DroneGameEnv(width=23, height=22)
        env = DroneGame(env=env, headless=args.headless)
        follower_env = FollowerWrapper(
            env=env, num_queries=2 ** env.observation_space("leader").n
        )
    follower_env = InfoSampleSingleAgentFollowerWrapper(follower_env)

    return follower_env


def pretrain(follower_env, config):
    checkpoints_path = f"checkpoints/{args.env}"
    follower_ckppath = os.path.join(checkpoints_path, "follower")

    if not os.path.exists(follower_ckppath):
        os.makedirs(follower_ckppath)

    latest_step = _latest_step(follower_ckppath) if os.listdir(follower_ckppath) else 0

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=follower_ckppath,
        name_prefix="ppo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    rmckp_callback = RmckpCallback(ckp_path=follower_ckppath)

    # Resuming
    if os.listdir(follower_ckppath):
        follower_model = PPO.load(
            os.path.join(follower_ckppath, f"ppo_{latest_step}_steps.zip"),
            env=follower_env,
        )
    # Starting from scratch
    else:
        follower_model = PPO(
            "MlpPolicy",
            follower_env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            **config,
        )

    leader_env = build_leader_env(
        follower_env=follower_env, follower_model=follower_model
    )

    leader_model = PPO("MlpPolicy", leader_env, verbose=1)

    callback_list = [checkpoint_callback, rmckp_callback]
    if args.log_wandb:
        wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        callback_list.append(wandb_callback)

    for _ in range(100):
        follower_env.update_leader_model(leader_model)
        follower_model.learn(
            total_timesteps=3000,
            reset_num_timesteps=False,
            callback=CallbackList(callback_list),
        )
        leader_env.update_follower_model(follower_model)
        leader_model.learn(total_timesteps=1000, reset_num_timesteps=False)
        leader_model.save(os.path.join(checkpoints_path, "leader_pretrained"))
    return follower_model


def test_pretrain(env):
    checkpoints_path = f"checkpoints/{args.env}/follower"
    latest_step = _latest_step(checkpoints_path)
    model = PPO.load(
        os.path.join(checkpoints_path, f"ppo_{latest_step}_steps.zip"),
        env=env,
    )
    if args.env == "matgame":
        for response in [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]:
            for s in range(5):
                obs = [s, *response]
                action = model.predict(obs, deterministic=True)[0]
                print(f"state: {s}, context: {response}, action: {action}")
    elif args.env == "dronegame":
        env.env.env.headless = False
        env.env.env.verbose = True
        leader_response = np.full((2**12,), 0, dtype=int)
        # leader_response = np.array(
        #     [0, 3, 0, 3, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3], dtype=int
        #     # [3, 1, 3, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 1], dtype=int
        # )
        obs, _ = env.reset(leader_response=leader_response)

        while True:
            action = model.predict(obs, deterministic=True)[0]
            new_obs, rewards, terminated, truncated, _ = env.step(action)
            print(obs, action, rewards)
            obs = new_obs

            if terminated or truncated:
                break

        env.env.env.env.close(video_name="size6_context_drone0.avi")


def build_leader_env(follower_env, follower_model=None):
    if follower_model is None:
        latest_step_follower = _latest_step(f"checkpoints/{args.env}/follower")
        follower_model = PPO.load(
            f"checkpoints/{args.env}/follower/ppo_{latest_step_follower}_steps.zip"
        )
    if args.env == "matgame":
        leader_env = SingleAgentLeaderWrapper(
            follower_env.env, queries=[0, 1, 2, 3, 4], follower_model=follower_model
        )
    else:
        num_queries = 2 ** follower_env.env.observation_space("leader").n
        queries = [
            [
                int(bit)
                for bit in np.binary_repr(
                    i, width=follower_env.env.observation_space("leader").n
                )
            ][::-1]
            for i in range(num_queries)
        ]
        leader_env = SingleAgentLeaderWrapper(
            follower_env.env, queries=queries, follower_model=follower_model
        )

    return leader_env


def train(leader_env):
    leader_ckppath = f"checkpoints/{args.env}/leader"
    if not os.path.exists(leader_ckppath):
        os.makedirs(leader_ckppath)

    if not os.listdir(leader_ckppath):
        leader_model = PPO.load(
            f"checkpoints/{args.env}/leader_pretrained.zip", env=leader_env
        )
    else:
        latest_step = _latest_step(leader_ckppath)
        leader_model = PPO.load(
            os.path.join(leader_ckppath, f"ppo_{latest_step}_steps.zip"), env=leader_env
        )

    if leader_model.tensorboard_log is None:
        leader_model.tensorboard_log = f"runs/{run.id}"

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=leader_ckppath,
        name_prefix="ppo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    rmckp_callback = RmckpCallback(ckp_path=leader_ckppath)

    callback_list = [checkpoint_callback, rmckp_callback]
    if args.log_wandb:
        wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        callback_list.append(wandb_callback)

    leader_model.learn(
        total_timesteps=200_000,
        callback=CallbackList(callback_list),
    )


def test_train(leader_env):
    leader_ckppath = f"checkpoints/{args.env}/leader"
    if (not os.path.exists(leader_ckppath)) or (not os.listdir(leader_ckppath)):
        model_path = f"checkpoints/{args.env}/leader_pretrained.zip"
    else:
        latest_step = _latest_step(leader_ckppath)
        model_path = os.path.join(leader_ckppath, f"ppo_{latest_step}_steps.zip")

    leader_model = PPO.load(model_path, env=leader_env)
    print("Loading model " + model_path + "\n")

    if args.env != "matgame":
        leader_env.env.env.headless = False
        leader_env.env.env.verbose = True
        leader_env.env.env.sleep_time = 0.7

    # play a single episode to check learned leader and follower policies
    obs, _ = leader_env.reset()
    while True:
        action = leader_model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = leader_env.step(action)
        print(obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break
    leader_env.env.env.env.close(video_name="complete_round.avi")


def print_policy():
    leader_model = PPO.load(f"checkpoints/leader_ppo_{args.env}")
    for o in range(16):
        o_bin = [int(bit) for bit in np.binary_repr(o, 4)][::-1]
        action = leader_model.predict(o_bin, deterministic=True)[0]
        print(f"obs: {o_bin}, act: {action}")


if __name__ == "__main__":
    follower_env = build_follower_env()

    pretrain_config = {
        "learning_rate": lambda progress: 0.001 * (1 - progress) + 0.00001 * progress,
        "gamma": 0.95,
        "ent_coef": 0.0,
        "batch_size": 128,
        "n_steps": 512,
    }

    if args.pretrain:
        pretrain(follower_env=follower_env, config=pretrain_config)
    if args.test_pretrain:
        test_pretrain(env=follower_env)

    leader_env = build_leader_env(follower_env=follower_env)
    if args.train:
        train(leader_env=leader_env)
    if args.test_train:
        test_train(leader_env=leader_env)
        print_policy()
