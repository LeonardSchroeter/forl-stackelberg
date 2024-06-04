from stable_baselines3 import PPO

from envs.matrix_game import IteratedMatrixGame
from envs.maze_design import Maze, MazeDesign
from wrappers.single_agent import (
    SingleAgentFollowerWrapper,
    SingleAgentQueryLeaderWrapper,
)
from wrappers.follower import FollowerWrapper

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="choose you environment: mat-game, maze-design. Default to maze-design",
)
parser.add_argument("--headless", help="disable GUI", action="store_true")
args = parser.parse_args()


def build_follower_env():
    if args.env == "mat-game":
        env_follower = FollowerWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
        )
    else:
        grid_size = 7
        max_steps = 4 * grid_size**2
        env_follower = FollowerWrapper(
            MazeDesign(
                (
                    Maze(
                        size=7,
                        agent_start_pos=(1, 1),
                        agent_start_dir=0,
                        agent_view_size=3,
                        max_steps=max_steps,
                    )
                ),
                headless=args.headless,
            ),
            num_queries=5,
        )
    env_follower = SingleAgentFollowerWrapper(env_follower)

    return env_follower


def pretrain(env):
    run_follower = wandb.init(project="stackelberg-ppo-follower", sync_tensorboard=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run_follower.id}")
    model.learn(
        total_timesteps=50_000,
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )
    model.save("checkpoints/follower_ppo")

    return model


def test_pretrain(env, model):
    for response in [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]:
        for s in range(5):
            obs = [s, *response]
            action = model.predict(obs, deterministic=True)[0]
            print(f"state: {s}, context: {response}, action: {action}")


def build_leader_env(follower_model):
    if args.env == "mat-game":
        env_leader = FollowerWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
        )
    else:
        grid_size = 7
        max_steps = 4 * grid_size**2
        env_leader = FollowerWrapper(
            MazeDesign(
                (
                    Maze(
                        size=7,
                        agent_start_pos=(1, 1),
                        agent_start_dir=0,
                        agent_view_size=3,
                        max_steps=max_steps,
                    )
                ),
                headless=args.headless,
            ),
            num_queries=5,
        )
    env_leader = SingleAgentQueryLeaderWrapper(
        env_leader,
        queries=[0, 1, 2, 3, 4],
        follower_model=follower_model,
    )

    return env_leader


def train(env_leader):
    run_leader = wandb.init(project="stackelberg-ppo-leader", sync_tensorboard=True)
    leader_model = PPO(
        "MlpPolicy", env_leader, verbose=1, tensorboard_log=f"runs/{run_leader.id}"
    )
    leader_model.learn(
        total_timesteps=50_000,
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )
    leader_model.save("checkpoints/leader_ppo")

    return leader_model


def test_train(env_leader, leader_model):
    # play a single episode to check learned leader and follower policies
    obs, _ = env_leader.reset()
    while True:
        action = leader_model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = env_leader.step(action)
        print(obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break


if __name__ == "__main__":
    follower_env = build_follower_env()
    follower_model = pretrain(env=follower_env)
    test_pretrain(env=follower_env, model=follower_model)

    leader_env = build_leader_env(follower_model=follower_model)
    leader_model = train(env_leader=leader_env)
    test_train(env_leader=leader_env, leader_model=leader_model)
