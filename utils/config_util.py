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
        default=None,
    )

    parser.add_argument("--no_initseg", action="store_true")
    parser.add_argument("--rl2_inner_outer", action="store_true")

    args = parser.parse_args()
    for key, value in vars(args).items():
        for key_config, value_config in config.items():
            if key in value_config.keys() and (value is not None):
                config[key_config][key] = value

    config["training"]["checkpoint_path"] = os.path.join(
        "checkpoints", config["env"]["name"], config["training"]["algo_name"]
    )

    for key in config.keys():
        config[key] = argparse.Namespace(**config[key])
    config = argparse.Namespace(**config)

    return config
