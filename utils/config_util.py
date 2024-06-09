import os

import yaml

import argparse


def load_config_args_overwrite(file=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        choices=[
            "matrix_game",
            "drone_game",
        ],
        default=None,
    )

    parser.add_argument("--no_initseg", action="store_true")
    parser.add_argument("--inner_outer", action="store_true")
    parser.add_argument("--leader_test_env", action="store_true")

    args = parser.parse_args()

    if file is None:
        file = f"configs/{args.algo}.yml"
    with open(file, "rb") as f:
        config = yaml.safe_load(f.read())

    for key, value in vars(args).items():
        for key_config, value_config in config.items():
            if key in value_config.keys() and (value is not None):
                config[key_config][key] = value

    for key in config.keys():
        config[key] = argparse.Namespace(**config[key])
    config = argparse.Namespace(**config)

    config.no_initseg = args.no_initseg
    config.inner_outer = args.inner_outer
    config.leader_test_env = args.leader_test_env
    config.training.checkpoint_path = os.path.join(
        "checkpoints", config.env.name, config.training.algo_name
    )

    return config
