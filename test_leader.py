from testing.test import test_leader
from utils.config_util import load_config

if __name__ == "__main__":
    config = load_config()
    test_leader(config)
