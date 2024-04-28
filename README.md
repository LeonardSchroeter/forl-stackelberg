# Foundations of Reinforcement Learning Project

## Setting up environment

```
conda env create -f environment.yml
conda activate forl
git clone https://github.com/YanzhenXiangRobotics/minigrid.git # minigrid/ is in .gitignore so feel free to clone it to current repo
cd minigrid
python3 -m pip install -e .
```
## Running a demo
```python3 stackerlberg/envs/maze_design.py```

Encouraged to use VSCode debugger.
