from stable_baselines3 import PPO
import numpy as np
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback

from envs.matrix_game import IteratedMatrixGame

# from envs.maze_design import Maze, MazeDesign
from envs.drone_game import DroneGame, DroneGameEnv
from wrappers.single_agent import (
    SingleAgentFollowerWrapper,
    SingleAgentLeaderWrapper,
)
from wrappers.follower import FollowerWrapper



parser = argparse.ArgumentParser()
parser.add_argument("--env", help="choose you environment: matgame, dronegame")
parser.add_argument("--headless", help="disable GUI", action="store_true")
parser.add_argument("--verbose", help="anable outputs", action="store_true")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--resume_pretrain", action="store_true")
parser.add_argument("--test_pretrain", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--resume_train", action="store_true")
parser.add_argument("--test_train", action="store_true")
args = parser.parse_args()

if args.pretrain or args.train:
    run = wandb.init(project="forl-stackerlberg", sync_tensorboard=True)

def build_follower_env():
    if args.env == "matgame":
        env_follower = FollowerWrapper(
            IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
            num_queries=5,
        )
    elif args.env == "dronegame":
        env = DroneGameEnv(agent_start_pos=(1, 10))
        env = DroneGame(env=env, headless=args.headless)
        env_follower = FollowerWrapper(
            env=env, num_queries=2 ** env.observation_space("leader").n
        )
    env_follower = SingleAgentFollowerWrapper(env_follower)

    return env_follower


def pretrain(env, config):
    if args.resume_pretrain:
        model = PPO.load(f"checkpoints/follower_ppo_{args.env}", env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            **config,
        )
    model.learn(
        total_timesteps=300_000,
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )
    model.save(f"checkpoints/follower_ppo_{args.env}")
    return model


def test_pretrain(env):
    model = PPO.load(f"checkpoints/follower_ppo_{args.env}")
    if args.env == "matgame":
        for response in [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]:
            for s in range(5):
                obs = [s, *response]
                action = model.predict(obs, deterministic=True)[0]
                print(f"state: {s}, context: {response}, action: {action}")
    elif args.env == "dronegame":
        env.env.env.headless = False
        env.env.env.verbose = True
        # leader_response = np.full((16,), 3, dtype=int)
        leader_response = np.array(
            [0, 3, 1, 2, 3, 1, 2, 0, 0, 1, 2, 3, 0, 2, 1, 3], dtype=int
        )
        obs, _ = env.reset(leader_response=leader_response)
        while True:
            action = model.predict(obs, deterministic=True)[0]
            new_obs, rewards, terminated, truncated, _ = env.step(action)
            print(obs, action, rewards)
            obs = new_obs

            if terminated or truncated:
                break


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
        env = DroneGameEnv(agent_start_pos=(1, 10))
        env = DroneGame(env=env, headless=args.headless)
        num_queries = 2 ** env.observation_space("leader").n
        env_leader = FollowerWrapper(env=env, num_queries=num_queries)
        queries = [
            [
                int(bit)
                for bit in np.binary_repr(i, width=env.observation_space("leader").n)
            ]
            for i in range(num_queries)
        ]
        env_leader = SingleAgentLeaderWrapper(
            env_leader, queries=queries, follower_model=follower_model
        )

    return env_leader


def train(env_leader):
    if args.resume_train:
        leader_model = PPO.load(f"checkpoints/leader_ppo_{args.env}")
    else:
        leader_model = PPO(
            "MlpPolicy", env_leader, verbose=1, tensorboard_log=f"runs/{run.id}"
        )
    env_leader.set_leader_response(leader_model)
    leader_model.learn(
        total_timesteps=200_000,
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )
    leader_model.save(f"checkpoints/leader_ppo_{args.env}")


def test_train(env_leader):
    leader_model = PPO.load(f"checkpoints/leader_ppo_{args.env}")
    if not env_leader.leader_response:
        env_leader.set_leader_response(leader_model)

    if args.env != "matgame":
        env_leader.env.env.headless = False
        env_leader.env.env.verbose = True

    # play a single episode to check learned leader and follower policies
    obs, _ = env_leader.reset()
    while True:
        action = leader_model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = env_leader.step(action)
        print(obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break


def print_policy():
    leader_model = PPO.load(f"checkpoints/leader_ppo_{args.env}")
    for o in range(16):
        o_bin = [int(bit) for bit in np.binary_repr(o, 4)]
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
        pretrain(env=follower_env, config=pretrain_config)
    if args.test_pretrain:
        test_pretrain(env=follower_env)

    leader_env = build_leader_env()
    if args.train:
        train(env_leader=leader_env)
    if args.test_train:
        test_train(env_leader=leader_env)
        print_policy()
