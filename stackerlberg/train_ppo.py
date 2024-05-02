from stable_baselines3 import PPO

import numpy as np

from envs.matrix_game import IteratedMatrixGame
# from envs.maze_design import Maze, MazeDesign
from envs.drone_game import DroneGame, DroneGameEnv
from wrappers.single_agent import (
    SingleAgentFollowerWrapper,
    SingleAgentLeaderWrapper,
)
from wrappers.follower import FollowerWrapper

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--env", 
help="choose you environment: matgame, dronegame")
parser.add_argument("--headless", help="disable GUI", action="store_true")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--testpretrain", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--testtrain", action="store_true")
args = parser.parse_args()

def build_follower_env():
    if args.env == "matgame":
        env_follower = FollowerWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
        )   
    elif args.env == "dronegame":
        grid_size = 20
        max_steps = 2 * grid_size
        env = DroneGameEnv(size=grid_size, max_steps=max_steps)
        env = DroneGame(env=env, headless=args.headless)
        env_follower = FollowerWrapper(env=env, num_queries=2**env.observation_space("leader").n)
    env_follower = SingleAgentFollowerWrapper(env_follower)

    return env_follower

def pretrain(env):
    
    run_follower = wandb.init(project="stackerlberg-follower", sync_tensorboard=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run_follower.id}")
    model.learn(total_timesteps=50_000, callback=WandbCallback(gradient_save_freq=100, verbose=2))
    model.save(f"checkpoints/follower_ppo_{args.env}")

def test_pretrain():
    model = PPO.load(f"checkpoints/follower_ppo_{args.env}")
    if args.env == "matgame":
        for response in [[1,1,1,1,1], [0,0,0,0,0], [0,0,0,1,1]]:
            for s in range(5):
                obs = [s, *response]
                action = model.predict(obs, deterministic=True)[0]
                print(f"state: {s}, context: {response}, action: {action}")
    else:
        pass

def build_leader_env():
    follower_model = PPO.load(f"checkpoints/follower_ppo_{args.env}")
    if args.env == "matgame":
        env_leader = FollowerWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
    )
        env_leader = SingleAgentLeaderWrapper(
        env_leader, queries=[0, 1, 2, 3, 4], follower_model=follower_model
    )
    else:
        grid_size = 20
        max_steps = 2 * grid_size
        env = DroneGameEnv(size=grid_size, max_steps=max_steps)
        env = DroneGame(env=env, headless=args.headless)
        num_queries = 2**env.observation_space("leader").n
        env_leader = FollowerWrapper(env=env, num_queries=num_queries)
        queries = [[int(bit) for bit in np.binary_repr(i, width=env.observation_space("leader").n)] for i in range(num_queries)]
        env_leader = SingleAgentLeaderWrapper(
            env_leader, queries=queries, follower_model=follower_model
        )

    return env_leader

def train(env_leader):

    run_leader = wandb.init(project="stackerlberg-leader", sync_tensorboard=True)
    leader_model = PPO("MlpPolicy", env_leader, verbose=1, tensorboard_log=f"runs/{run_leader.id}")
    leader_model.learn(total_timesteps=50_000, callback=WandbCallback(gradient_save_freq=100, verbose=2))
    leader_model.save(f"checkpoints/leader_ppo_{args.env}")

def test_train(env_leader):

    leader_model = PPO.load(f"checkpoints/leader_ppo_{args.env}")

    if args.env != "matgame":
        env_leader.env.env.headless = False

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
    if args.pretrain:
        pretrain(env=follower_env)
    if args.testpretrain:
        test_pretrain()

    leader_env = build_leader_env()
    if args.train:
        train(env_leader=leader_env)
    if args.testtrain:
        test_train(env_leader=leader_env)
    

    

   

   
