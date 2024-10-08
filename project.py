import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Simple3DFlyer(gym.Env):
    def __init__(self):
        super(Simple3DFlyer, self).__init__()
        
        # Define action and observation space
        # Actions: [delta_x, delta_y, delta_z]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Observations: [x, y, z]
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)
        
        # Define start, goal, and obstacle
        self.start = np.array([0, 0, 0])
        self.goal = np.array([100, 100, 100])
        self.state = self.start.copy()
        self.obstacles = [
            {'location': np.array([50, 50, 50]), 'radius': 10}
        ]
        self.done = False

    def reset(self):
        self.state = self.start.copy()
        self.done = False
        return self.state
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done")
        
        # Apply action to state
        self.state = self.state + action
        
        # Enforce boundary limits
        self.state = np.clip(self.state, 0, 100)
        
        # Check for reaching the goal
        if np.linalg.norm(self.state - self.goal) < 5:
            self.done = True
            reward = 100  # reward for reaching the goal
        else:
            # Check for collisions with obstacles
            reward = -1
            for obs in self.obstacles:
                if np.linalg.norm(self.state - obs['location']) < obs['radius']:
                    reward = -100  # penalty for collision
                    self.done = True
                    break
                    
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        # Rendering logic, if any (could just print state)
        print(f"Current state: {self.state}")

    def close(self):
        pass

# Usage
env = Simple3DFlyer()
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    state, reward, done, _ = env.step(action)
    env.render()