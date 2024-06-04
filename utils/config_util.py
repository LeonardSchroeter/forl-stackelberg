import os

import yaml

import argparse


def load_config_args_overwrite(file):
    with open(file, "rb") as file:
        config = yaml.safe_load(file.read())

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        choices=[
            "bandit",
            "tabular_mdp",
            "matrix_game",
            "drone_game",
        ],
        default="bandit",
    )

    args = parser.parse_args()
    for key, value in vars(args).items():
        for key_config, value_config in config.items():
            if key in value_config.keys():
                config[key_config][key] = value

    config["training"]["checkpoint_path"] = os.path.join(
        f"checkpoints/{config["env"]["name"]}", config["training"]["algo_name"]
    )

    for key in config.keys():
        config[key] = argparse.Namespace(**config[key])
    config = argparse.Namespace(**config)

    return config
