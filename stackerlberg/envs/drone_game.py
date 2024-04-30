from __future__ import annotations

from typing import List

import time

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces

from pettingzoo import ParallelEnv
        

class DroneGameEnv(MiniGridEnv):
    def __init__(
        self,
        size=20,
        agent_start_pos=(1, 10),
        agent_start_dir=0,
        agent_view_size=3,
        max_steps: int | None = None,
        drone_options: List = [(13, 4), (13, 8), (13, 12), (13, 16)],
        num_divisions=4,
        drone_cover_size=3,
        
        **kwargs,
    ): 
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_view_size = agent_view_size
        self.drone_options = drone_options
        self.num_divisions = num_divisions
        self.drone_cover_size = drone_cover_size

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
        return "drone game"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        for j in range(1, self.width - 1):
            self.put_obj(Goal(), self.height - 2, j)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "drone game"

    

class DroneGame(ParallelEnv):

    metadata = {
        "name": "drone_game",
    }

    def __init__(self, env: DroneGameEnv, headless: bool = False) -> None:
        super().__init__()

        self.env = env
        self.headless = headless
        if not headless:
            self.env.render_mode = "human"

        self.agents = ["leader", "follower"]

        agent_view_area = env.agent_view_size * env.agent_view_size
        # grid_area = (env.width - 2) * (env.height - 2) # The outer are walls so -2

        # leader action: which of prescribed places to place drone
        # follower action: fwd(0), left(1), right(2)
        self.action_spaces = {
            "leader": spaces.Discrete(len(self.env.drone_options)),
            "follower": spaces.Discrete(3),
        }
        # leader observation: which division does the follower lie in
        # follower observation: wall occupancy in its local view size
        self.observation_spaces = {
            "leader": spaces.Discrete(2**self.env.num_divisions),
            "follower": spaces.Discrete(2**agent_view_area),
        }

        self.drones = []

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def reset(self):
        self.env.reset()
        return {"leader": 0, "follower": 0}
    
    def step(self, actions):
        
        print(f"\n STEP: {self.env.step_count}\n")

        observations, rewards = {}, {}

        if len(self.drones) >= 4:
            dead_drone = self.drones.pop(0)
            dead_drone.undo_lava()
            del dead_drone

        if not self.headless:
            self.env.render()
            time.sleep(0.5)

        print("Leader takes action")
        self.leader_act(actions["leader"])
        observations["leader"] = self.get_leader_observation() 
        rewards["leader"] = self.get_leader_reward()

        print(f"leader observation: {observations["leader"]}")
        print(f"leader reward: {rewards["leader"]}")


        for drone in self.drones:
            drone.browian_motion()

        if not self.headless:
            self.env.render()
            time.sleep(0.5)

        print("Follower takes action")
        self.follower_act(actions["follower"])
        if not self.headless:
            self.env.render()
            time.sleep(0.5)
        observations["follower"] = self.get_follower_observation() 
        rewards["follower"] = -rewards["leader"] + self.env.height

        print(f"follower observation: {observations["follower"]}")
        print(f"follower observation binary: {np.binary_repr(observations["follower"])}")
        print(f"follower reward: {rewards["follower"]}")
        
        terminated = (self.env.step_count >= self.env.max_steps) or \
        (self.env.agent_pos[0] == self.env.height - 2) or \
        isinstance(self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]), Lava)

        print(f"terminated: {terminated}")

        return observations, rewards, terminated, {}, {}

    def leader_act(self, action):
        
        drone_place = self.env.drone_options[action]
        self.drones.append(Drone(env=self.env, size=3, center=drone_place))

    def binary_matrix_to_int(self, binary_matrix: np.array):
        binary_vec = binary_matrix.T.flatten()
        binary_str = "".join(str(bit) for bit in binary_vec)[::-1]
        int_number = int(binary_str, base=2)
        return int_number

    def get_leader_observation(self):

        m = (self.env.agent_pos[0] < self.env.width / 2,
             self.env.agent_pos[1] < self.env.height / 2)
        if m == (1, 1):
            observation = 0
        elif m == (1, 0):
            observation = 1
        elif m == (0, 1):
            observation = 2
        elif m == (0, 0):
            observation = 3 

        return observation
    
    def get_leader_reward(self):
        
        reward = self.env.height - 1 - self.env.agent_pos[0]

        if isinstance(self.env.grid.get(self.env.agent_pos[0], 
                                        self.env.agent_pos[1]),
                      Lava):
            reward += self.env.height
        
        if isinstance(self.env.grid.get(self.env.agent_pos[0], 
                                        self.env.agent_pos[1]),
                     Goal):
            reward -= self.env.height

        return reward

    def follower_act(self, action):
        match action:
            case 0: #fwd
                self.env.step(self.env.actions.forward)
            # case 1: #bwd
            #     self.env.step(self.env.actions.backward)
            case 1: #left
                self.env.step(self.env.actions.move_left)
            case 2: #right
                self.env.step(self.env.actions.move_right)

    def get_follower_observation(self):

        topX, topY, botX, botY = self.env.get_view_exts()
        
        local_lava = np.full((self.env.agent_view_size, self.env.agent_view_size), 0)
        for i in range(topX, botX):
            for j in range(topY, botY):
                i_local, j_local = self.env.relative_coords(i, j)
                if not Drone.in_grid(i, j, self.env.width, self.env.height): # Out of grid map: deemed as lava (TODO: improve)
                    local_lava[i_local, j_local] = 1
                else:
                    tile = self.env.grid.get(i, j)
                    if isinstance(tile, Lava):
                        local_lava[i_local, j_local] = 1
                
        observation = self.binary_matrix_to_int(local_lava)
        
        return observation
    
class Drone:
    def __init__(self, env: DroneGameEnv, center, size=3) -> None:
        self.size = size
        # self.to_death = lifespan
        self.center = center
        self.env = env
        self.set_lava()

    @staticmethod
    def in_grid(i, j, width, height):
        if (i >= 1) and (i < width - 1) and (j >= 1) and (j < height - 1):
            return True
        return False

    def set_lava(self):
        for i in range(int(self.center[0] - (self.size - 1) / 2), 
                    int(self.center[0] + (self.size - 1) / 2)):
            for j in range(int(self.center[1] - (self.size - 1) / 2), 
                    int(self.center[1] + (self.size - 1) / 2)):
                if self.in_grid(i, j, self.env.width, self.env.height):
                    self.env.put_obj(Lava(), i, j)

    def undo_lava(self):
        for i in range(int(self.center[0] - (self.size - 1) / 2), 
                    int(self.center[0] + (self.size - 1) / 2)):
            for j in range(int(self.center[1] - (self.size - 1) / 2), 
                    int(self.center[1] + (self.size - 1) / 2)):
                if self.in_grid(i, j, self.env.height, self.env.height):
                    self.env.grid.set(i, j, None)

    
    def browian_motion(self):

        # self.to_death -= 1

        self.undo_lava()

        # if self.to_death > 0:
        while True:
            dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            move_dir = np.random.randint(4)
            m = (self.size - 1) / 2
            next_center = (self.center[0] + dirs[move_dir][0], self.center[1] + dirs[move_dir][1])
            if self.in_grid(next_center[0] - m, next_center[1] - m, 
                            width=self.env.width, height=self.env.height) and \
            self.in_grid(next_center[0] - m, next_center[1] + m,
                            width=self.env.width, height=self.env.height) and \
            self.in_grid(next_center[0] + m, next_center[1] - m,
                            width=self.env.width, height=self.env.height) and \
            self.in_grid(next_center[0] + m, next_center[1] + m,
                            width=self.env.width, height=self.env.height): 
                self.center = next_center
                break

        self.set_lava()

    # def get_follower_reward(self):
    #     pass

if __name__ == "__main__":
    env = DroneGameEnv(agent_start_pos=(3,10))
    env = DroneGame(env=env)

    actions= {}
    actions["follower"] = 0

    i = 0
    env.env.reset()
    while True:
        actions["leader"] = i % 4
        observation, reward, terminated, _, _ = env.step(actions)
        if terminated:
            break
        i += 1
        
