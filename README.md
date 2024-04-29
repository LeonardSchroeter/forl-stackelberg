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

`python3 stackerlberg/envs/maze_design.py`

Encouraged to use VSCode debugger.

## Security Game

### Leader

Each step, the leader receives a deployable drone with probability p (we could think about making this easier by setting this p = 1). If the leader receives a drone, he can choose on which out of several pre-defined positions to deploy the drone. For example, there could be a few positions on a line dividing the grid into two regions. Additionally, the grid is divided into several regions by horizontal and vertical lines, and the leader observes for each region wether ones of his drones is inside this region or not. The state space of the leader hence becomes

`MultiDiscrete(2, {2}^n)`

where the first entry indicates if the leader can deploy a drone, and the following n entries indicate for each region if a drone is inside or not. The action space of the leader is

`Discrete(m)`

where m is the number of possible deployment positions.

### Follower

The follower either observes the positions of all drones, i.e. for each position on the grid, he observes if a drone is there or not, or he does not observe the positions of the drones. In that case, the state space of the follower would just contain a single state. Alternatively, the follower could observe the same information about drone positions as the leader. The action space of the follower is just the set of all directions he can move.
