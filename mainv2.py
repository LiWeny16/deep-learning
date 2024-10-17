import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from gymnasium import spaces
import pickle

class UAV3DEnv(gym.Env):
    def __init__(self, grid_size=(11, 40, 11), goal_position=(5, 39, 5), uav_radius=0.5, obstacle_radius=0.5):
        super().__init__()
        self.grid_size = grid_size
        self.uav_radius = uav_radius
        self.obstacle_radius = obstacle_radius
        
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(3,), dtype=np.int32)
        self.action_space = spaces.Discrete(5)

        self.state = np.array([0, 0, 0], dtype=np.int32)
        self.goal_position = np.array(goal_position, dtype=np.int32)
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < 20:
            obstacle = tuple(np.random.randint(0, [self.grid_size[0], self.grid_size[1], self.grid_size[2]]))
            if obstacle != tuple(self.state) and obstacle != tuple(self.goal_position):
                obstacles.add(obstacle)
        return list(obstacles)

    def reset(self):
        self.state = np.array([0, 0, 0], dtype=np.int32)
        self.obstacles = self._generate_obstacles()
        self.previous_states = [self.state.copy()] * 7  # Initialize the last 7 positions
        return self.state, {}

    def step(self, action):
        moves = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
        new_state = self.state + moves[action]
        
        # Check for boundary crossing
        if np.any(new_state < 0) or np.any(new_state >= np.array(self.grid_size)):
            reward = -50  # Penalty for trying to cross the boundary
            terminated = False
            new_state = self.state  # Stay in the current state
        else:
            self.state = new_state
            reward = -2
            terminated = False
            if tuple(self.state) in self.obstacles:
                reward = -30  # Lesser penalty
                terminated = True  # Allows continued attempt to select new action
            elif np.array_equal(self.state, self.goal_position):
                reward = 100
                terminated = True

        distance_to_goal = np.linalg.norm(self.state - self.goal_position)
        if np.linalg.norm(new_state - self.goal_position) < distance_to_goal:
            reward += 5  # Additional reward for moving towards the goal
        # Update the last 7 positions
        self.previous_states.pop(0)
        self.previous_states.append(self.state.copy())

        # Check for hovering in place
        if self._is_hovering():
            reward = -10  # Extra penalty

        return self.state, reward, terminated, False, {}

    def _is_hovering(self):
        # Check if the last 7 positions are within a small range
        recent_positions = np.array(self.previous_states)
        max_distance = np.max(np.ptp(recent_positions, axis=0))
        return max_distance < 4  # Considered hovering if the maximum distance is less than 4
    def render(self):
        plt.clf()
        ax = plt.subplot(111, projection='3d')

        self._draw_sphere(ax, self.state, self.uav_radius, 'blue', 'UAV')
        self._draw_sphere(ax, self.goal_position, self.uav_radius, 'green', 'Goal')
        for obs in self.obstacles:
            self._draw_sphere(ax, obs, self.obstacle_radius, 'red')

        # Set axis ranges to fit the grid size
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_zlim(0, self.grid_size[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set the image scale to fit the grid size
        ax.set_box_aspect([self.grid_size[0], self.grid_size[1], self.grid_size[2]])
        plt.pause(0.1)

    def _draw_sphere(self, ax, center, radius, color, label=None):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=0.3, label=label)

def q_learning_train(env, episodes=5000, alpha=0.25, gamma=0.85, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    q_table = np.zeros((*env.grid_size, env.action_space.n))
    success_rate = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        success = 0
        print("Episode:", episode)
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[tuple(state)])
            
            next_state, reward, done, _, _ = env.step(action)
            
            q_table[tuple(state)][action] += alpha * (reward + gamma * np.max(q_table[tuple(next_state)]) - q_table[tuple(state)][action])
            state = next_state
            success = int(reward == 100)
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        success_rate.append(success)

    plt.plot(np.convolve(success_rate, np.ones(20)/20, mode='valid'))
    plt.title('Success Rate per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.show()
    
    with open("./q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

def q_learning_visualize(env):
    with open("./q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    
    state, _ = env.reset()
    done = False
    
    while not done:
        action = np.argmax(q_table[tuple(state)])
        state, _, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.1)

env = UAV3DEnv()

# Training mode
q_learning_train(env)

# Visualization mode
plt.ion()
q_learning_visualize(env)
plt.ioff()
plt.show()
