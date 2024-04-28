from __future__ import annotations

import time

import numpy as np
from bitstring import BitArray

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import highlight_img

from gymnasium import spaces

from pettingzoo import ParallelEnv

class Maze(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        agent_view_size=3,
        max_steps: int | None = None,
        **kwargs,
    ): 
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_view_size = agent_view_size
        self.path = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "maze"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.goal_pos = (width - 2, 1)
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "maze"

class MazeDesign(ParallelEnv):

    metadata = {
        "name": "maze_design",
    }

    def __init__(self, env: Maze, episode_length: int = 1) -> None:
        super().__init__()

        self.env = env

        self.agents = ["leader", "follower"]

        agent_view_area = env.agent_view_size * env.agent_view_size
        grid_area = (env.width - 2) * (env.height - 2) # The outer are walls so -2
        
        # design size: size (side length) of local area for leader to design at each timestep
        self.design_size = env.agent_view_size
        self.design_area = self.design_size**2

        # leader action: whether each tile in the area to design is placed wall or not
        # follower action: rotate ccw(0), rotate cw(1), move forward(2)
        self.action_spaces = {
            "leader": spaces.Discrete(2**self.design_area),
            "follower": spaces.Discrete(3),
        }
        # leader observation: agent_pos, agent_dir, wall_occupancy
        # follower observation: local_goal_pos(-1 if not in agent's view), agent_dir, local_wall_occupancy
        self.observation_spaces = {
            "leader": spaces.MultiDiscrete([grid_area, 4, 2**grid_area]),
            "follower": spaces.MultiDiscrete([agent_view_area + 1, 4, 2**agent_view_area]),
        }

        self.episode_length = episode_length
        self.current_step = 0

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def reset(self):
        # self.current_step = 0
        return {"leader": 0, "follower": 0}
    
    def get_colors(self):
        colors = np.zeros((self.env.width, self.env.height))
        for i in range(self.env.width):
            for j in range(self.env.height):
                colors = self.env.grid.get(i, j).color

        return colors
    
    def set_colors(self, colors):
        for i in range(self.env.width):
            for j in range(self.env.height):
                self.env.grid.get(i, j).color = colors[i, j]
    
    def step(self, actions):

        observations, rewards = {}, {}

        self.current_step += 1

        print("Leader takes action")
        self.leader_act(actions["leader"])
        # leader observation: agent_pos, agent_dir, wall_occupancy
        observations["leader"] = self.get_leader_observation() 
        rewards["leader"] = self.get_leader_reward()

        print(f"leader observation: {observations["leader"]}")
        print(f"global wall binary: {np.binary_repr(observations["leader"][-1])}")
        print(f"leader reward: {rewards["leader"]}")

        self.env.render()
        time.sleep(0.5)

        print("Follower takes action")
        # follower observation: local_goal_pos, agent_dir, local_wall_occupancy
        observations["follower"], rewards["follower"] = self.follower_step(actions["follower"])

        print(f"follower observation: {observations["follower"]}")
        print(f"local wall binary: {np.binary_repr(observations["follower"][-1])}")
        print(f"follower reward: {rewards["follower"]}")

        self.env.render()
        time.sleep(0.5)
        
        terminated = False
        if self.env.agent_pos == self.env.goal_pos:
            terminated = True

        print(f"terminated: {terminated}")

        return observations, rewards, terminated

    def in_grid(self, i, j):
        if (i >= 1) and (i < self.env.width - 1) and (j >= 1) and (j < self.env.height - 1):
            return True
        return False

    def leader_act(self, action):
        
        local_wall_design = np.reshape([int(bit) for bit in np.binary_repr(action, 
                                width=self.design_area)][::-1], 
                                newshape=(self.design_size, self.design_size))
        
        topX, topY, botX, botY = self.env.get_view_exts()
        
        for i in range(topX, botX):
            for j in range(topY, botY):
                if self.in_grid(i, j):
                    if (i, j) != self.env.agent_pos:
                        if local_wall_design[j - topY][i - topX] == 1:
                            self.env.put_obj(Wall(), i, j)

    def binary_matrix_to_int(self, binary_matrix: np.array):
        binary_vec = binary_matrix.T.flatten()
        binary_str = "".join(str(bit) for bit in binary_vec)[::-1]
        int_number = int(binary_str, base=2)
        return int_number

    def get_leader_observation(self):
        wall_design = np.full((self.env.width - 2, self.env.height - 2),0)
        for i in range(1, self.env.width - 1):
            for j in range(1, self.env.height - 1):
                tile = self.env.grid.get(i, j)
                if isinstance(tile, Wall):
                    wall_design[i - 1, j - 1] = 1
        wall_design_observation = self.binary_matrix_to_int(wall_design)
        
        observation = np.array([(self.env.agent_pos[1] - 1) * (self.env.width - 2) + \
        (self.env.agent_pos[0] - 1), self.env.agent_dir, wall_design_observation])
        return observation
    
    def get_leader_reward(self):
        
        open_set = [self.env.agent_pos]
        parents = {}
        g_score = {self.env.agent_pos: 0}

        while open_set:
            curr_pos = min(open_set, key=lambda pos: g_score.get(pos, float('inf')))

            if curr_pos == self.env.goal_pos:
                break

            open_set.remove(curr_pos)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (curr_pos[0] + dx, curr_pos[1] + dy)
                if self.in_grid(neighbor[0], neighbor[1]) and \
                   (not isinstance(self.env.grid.get(neighbor[0], neighbor[1]), Wall)):
                    g_tentative = g_score[curr_pos] + 1
                    if g_tentative < g_score.get(neighbor, float('inf')):
                        parents[neighbor] = curr_pos
                        g_score[neighbor] = g_tentative
                        if neighbor not in open_set:
                            open_set.append(neighbor)

        path = []
        while curr_pos != self.env.agent_pos:
            path.append(curr_pos)
            curr_pos = parents[curr_pos]
        path.reverse()

        self.env.path = path
        if self.env.goal_pos not in path:
            self.env.path = []
            return - self.env.width * self.env.height
        return len(path)

    def follower_step(self, action):
        _, reward, _, _, _ = self.env.step(action)

        topX, topY, botX, botY = self.env.get_view_exts()
        
        local_walls = np.full((self.env.agent_view_size, self.env.agent_view_size), 0)
        for i in range(topX, botX):
            for j in range(topY, botY):
                i_local, j_local = self.env.relative_coords(i, j)
                if not self.in_grid(i, j): # Out of grid map: deemed as walls
                    local_walls[i_local, j_local] = 1
                else:
                    tile = self.env.grid.get(i, j)
                    if isinstance(tile, Wall):
                        local_walls[i_local, j_local] = 1
                
        wall_design_observation = self.binary_matrix_to_int(local_walls)
        
        local_goal_pos = self.env.relative_coords(self.env.goal_pos[0], self.env.goal_pos[1])
        if local_goal_pos is None:
            local_goal_pos_index = -1
        else:
            local_goal_pos_index = local_goal_pos[1] * self.env.agent_view_size + local_goal_pos[0]
        observation = np.array([local_goal_pos_index, self.env.agent_dir, wall_design_observation])

        return observation, reward

if __name__ == "__main__":
    env = Maze(size=9, agent_start_pos=(1, 1), agent_start_dir=0, agent_view_size=3)
    env = MazeDesign(env=env, episode_length=20)

    env.env.reset()

    actions= {}
    actions["leader"] = 2**8
    actions["follower"] = 2
    env.env.render_mode = "human"

    # for i in range(env.env.width):
    #         for j in range(env.env.height):
                # print(env.env.grid.get(i, j))

    for _ in range(7):
        
        # if hasattr(env, "colors"):
        #     env.set_colors(env.colors)
        #     env.env.render()

        observation, reward, terminated = env.step(actions)
        
        
        # print(reward)
        # print(f"observation: {observation}, reward: {reward}, terminated: {terminated}")
        # env.env.step(2)
    while True:
        env.env.render()
        
