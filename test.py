import gymnasium as gym
import numpy as np
from gymnasium import spaces

class UAV3DEnv(gym.Env):
    def __init__(self, grid_size=(10, 10, 10), goal_position=(9, 9, 9)):
        super(UAV3DEnv, self).__init__()
        
        self.grid_size = grid_size
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(3,), dtype=np.int32)
        self.action_space = spaces.Discrete(6)  # 上下左右前后
        
        self.state = np.array([0, 0, 0], dtype=np.int32)
        self.goal_position = np.array(goal_position, dtype=np.int32)
        self.obstacles = self._generate_obstacles()
    
    def _generate_obstacles(self):
        obstacles = []
        for _ in range(20):  # 生成20个随机障碍物
            while True:
                obstacle = np.random.randint(0, max(self.grid_size), size=3)
                if not np.array_equal(obstacle, self.goal_position) and not np.array_equal(obstacle, self.state):
                    obstacles.append(tuple(obstacle))
                    break
        return obstacles
    
    def reset(self):
        self.state = np.array([0, 0, 0], dtype=np.int32)
        return self.state, {}
    
    def step(self, action):
        if action == 0:  # 向上
            self.state[0] = min(self.state[0] + 1, self.grid_size[0] - 1)
        elif action == 1:  # 向下
            self.state[0] = max(self.state[0] - 1, 0)
        elif action == 2:  # 向前
            self.state[1] = min(self.state[1] + 1, self.grid_size[1] - 1)
        elif action == 3:  # 向后
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 4:  # 向左
            self.state[2] = min(self.state[2] + 1, self.grid_size[2] - 1)
        elif action == 5:  # 向右
            self.state[2] = max(self.state[2] - 1, 0)

        reward = -1
        terminated = False
        if tuple(self.state) in self.obstacles:
            reward = -10  # 撞到障碍物
            terminated = True
        elif np.array_equal(self.state, self.goal_position):
            reward = 100  # 达到目标
            terminated = True
        
        return self.state, reward, terminated, False, {}
        
    def render(self):
        grid = np.full(self.grid_size, '.', dtype=str)
        
        grid[tuple(self.state)] = 'U'
        grid[tuple(self.goal_position)] = 'G'
        
        for obs in self.obstacles:
            grid[obs] = 'X'
        
        for layer in grid:
            for row in layer:
                print(" ".join(row))
            print("\n")
    
env = UAV3DEnv()
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, _, info = env.step(action)
    
    env.render()
    
    if terminated:
        observation, info = env.reset()

env.close()