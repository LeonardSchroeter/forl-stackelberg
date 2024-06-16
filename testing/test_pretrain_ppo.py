import os

import numpy as np

from training.ppo.pretrain import build_follower_env
from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config_args_overwrite

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["ppo", "rl2"], default="rl2")


def test_pretrain(config, checkpoint_path):
    env = build_follower_env(config)

    model, _ = maybe_load_checkpoint_ppo(checkpoint_path, env)

    if config.env.name == "matrix_game":
        for response in [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]:
            for s in range(5):
                obs = [s, *response]
                action = model.predict(obs, deterministic=True)[0]
                print(f"state: {s}, context: {response}, action: {action}")
    elif config.env.name == "drone_game":
        env.plant.headless = False
        # leader_response = np.full((2**10,), 9, dtype=int)
        leader_response = np.full((11**4,), 0, dtype=float)
        print(len(leader_response))
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

        env.plant.close(video_name=config.drone_game.video_name)


if __name__ == "__main__":
    config = load_config_args_overwrite(parser=parser)
    if config.inner_outer:
        folder = "inner_outer"
    elif config.env.name == "drone_game" and config.drone_game.leader_cont:
        folder = "leader_cont"
    else:
        folder = ""
    test_pretrain(
        config,
        checkpoint_path=os.path.join(
            config.training.checkpoint_path, folder, "follower"
        ),
    )
