from setuptools import setup, find_packages

setup(
    name='forl-stackelberg',
    version='0.1',
    packages=find_packages(include=["envs", "rl2_agents", "testing", "training", "tuning", "wrappers", "algos", "utils"])
)
