import yaml

import argparse


def dict_to_namespace(d):
    """
    Recursively converts a nested dictionary to an argparse.Namespace object.

    Parameters:
    d (dict): The dictionary to convert.

    Returns:
    argparse.Namespace: The resulting Namespace object.
    """
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively convert the nested dictionary
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def load_config(config_name: str | None = None):
    parser = argparse.ArgumentParser()

    if not config_name:
        parser.add_argument("--config", type=str, default=None, required=True, help="The config file to use.")

    args = parser.parse_args()

    config_path = f"configs/{config_name or args.config}.yml"
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f.read())

    return dict_to_namespace(config)
