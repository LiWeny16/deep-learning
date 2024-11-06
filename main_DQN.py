import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

# Define the UAV 3D Environment
class UAV3DEnv(gym.Env):
    def __init__(self, grid_size=(11, 40, 11), goal_position=(5, 39, 5)):
        super(UAV3DEnv, self).__init__()
        self.grid_size = np.array(grid_size)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32)
        self.action_space = spaces.Discrete(5)
        self.goal_position = np.array(goal_position, dtype=np.int32)
        self.obstacles = self._generate_obstacles()
        self.reset()

    def _generate_obstacles(self):
        obstacles = {
            (2, 10, 0), (2, 11, 0), (2, 12, 0), (2, 10, 1), (2, 11, 1), (2, 12, 1),
            (2, 10, 2), (2, 11, 2), (2, 12, 2), (2, 10, 3), (2, 11, 3), (2, 12, 3),
            (8, 10, 0), (8, 11, 0), (8, 12, 0), (8, 10, 1), (8, 11, 1), (8, 12, 1),
            (8, 10, 2), (8, 11, 2), (8, 12, 2), (8, 10, 3), (8, 11, 3), (8, 12, 3),
            (4, 30, 0), (5, 30, 0), (6, 30, 0), (7, 30, 0), (4, 30, 1), (5, 30, 1),
            (6, 30, 1), (7, 30, 1), (4, 30, 2), (5, 30, 2), (6, 30, 2), (7, 30, 2),
            (4, 20, 0), (6, 20, 0), (4, 20, 1), (6, 20, 1), (4, 20, 2), (6, 20, 2),
            (4, 20, 3), (6, 20, 3), (4, 20, 4), (6, 20, 4), (4, 20, 5), (6, 20, 5),
        }
        # obstacles2 = {(2,10,0),(2,11,0),(2,12,0),(2,10,1),(2,11,1),(2,12,1),(2,10,2),(2,11,2),(2,12,2),(2,10,3),(2,11,3),(2,12,3),
        #              (8,10,0),(8,11,0),(8,12,0),(8,10,1),(8,11,1),(8,12,1),(8,10,2),(8,11,2),(8,12,2),(8,10,3),(8,11,3),(8,12,3),
        #              (4,30,0),(5,30,0),(6,30,0),(7,30,0), (4,30,1),(5,30,1),(6,30,1),(7,30,1),(4,30,2),(5,30,2),(6,30,2),(7,30,2),
        #              (4,20,0),(6,20,0),(4,20,1),(6,20,1), (4,20,0),(4,20,2),(6,20,2),(4,20,3),(6,20,3),(4,20,4),(6,20,4),(4,20,5),(6,20,5),
        #              (0,3,0),(1,3,0),(2,3,0),(3,3,0),(4,3,0),(5,3,0),(5,3,0),(6,3,0),(7,3,0),(8,3,0),(9,3,0),(10,3,0),(11,3,0),
        #              (0,4,0),(1,4,0),(2,4,0),(3,4,0),(4,4,0),(5,4,0),(5,4,0),(6,4,0),(7,4,0),(8,4,0),(9,4,0),(10,4,0),(11,4,0),
        #              (5,3,1),(5,4,1),(5,3,2),(5,4,2),(5,3,3),(5,4,3),(5,3,4),(5,4,4),(5,3,5),(5,4,5),(5,3,6),(5,4,6),(5,3,7),(5,4,7),
        #              }
        return obstacles

    def reset(self):
        self.state = np.array([5, 0, 0], dtype=np.int32)
        self.previous_states = [self.state.copy()] * 5
        self.history = [self.state.copy()]
        return self.state.copy(), {}

    def step(self, action):
        moves = np.array([
            [1, 0, 0],    # Right
            [-1, 0, 0],   # Left
            [0, 1, 0],    # Forward
            [0, 0, 1],    # Up
            [0, 0, -1]    # Down
        ])
        new_state = self.state + moves[action]
        terminated = False

        # Check for out-of-bounds
        if np.any(new_state < 0) or np.any(new_state >= self.grid_size):
            reward = -50
            new_state = self.state.copy()
        else:
            self.state = new_state.copy()
            if tuple(self.state) in self.obstacles:
                reward = -100
                terminated = True
            elif np.array_equal(self.state, self.goal_position):
                reward = 500
                terminated = True
            else:
                # Negative Manhattan distance to the goal
                reward = -np.sum(np.abs(self.state - self.goal_position))

        # Update previous states and history
        self.previous_states.pop(0)
        self.previous_states.append(self.state.copy())
        self.history.append(self.state.copy())

        # Additional penalty for hovering
        if self._is_hovering():
            reward -= 30

        return self.state.copy(), reward, terminated, False, {}

    def _is_hovering(self):
        recent_positions = np.array(self.previous_states)
        max_distance = np.max(np.ptp(recent_positions, axis=0))
        return max_distance < 2

    def render(self):
        plt.clf()
        ax = plt.subplot(111, projection='3d')
        self._draw_point(ax, self.state, 'blue')
        self._draw_point(ax, self.goal_position, 'green')
        for obs in self.obstacles:
            self._draw_point(ax, np.array(obs), 'red')
        # Plot trajectory
        history = np.array(self.history)
        ax.plot(history[:, 0], history[:, 1], history[:, 2], color='blue', marker='o', markersize=2,
                linestyle='-', linewidth=1, alpha=0.6)
        # Set axis limits
        ax.set_xlim(0, self.grid_size[0] - 1)
        ax.set_ylim(0, self.grid_size[1] - 1)
        ax.set_zlim(0, self.grid_size[2] - 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect(self.grid_size)
        plt.pause(0.01)

    def _draw_point(self, ax, center, color):
        ax.scatter(center[0], center[1], center[2], color=color, s=100)

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean()
        return q_values

# Define the Neural Network Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Deep Q-Learning Algorithm
def train_dqn(env, num_episodes=1000, batch_size=64, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500,
              target_update=10, memory_capacity=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayBuffer(memory_capacity)

    steps_done = 0
    episode_durations = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        total_reward = 0
        done = False

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1

            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.max(0)[1].item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state).to(device)
            memory.push(state.cpu().numpy(), action, reward, next_state, done)

            state = next_state_tensor

            # Perform one step of the optimization
            if len(memory) >= batch_size:
                optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device)

        # Update the target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_durations.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Plot the training progress
    plt.figure()
    plt.plot(episode_durations)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Save the trained model
    torch.save(policy_net.state_dict(), 'dqn_model.pth')

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

    # Compute Q(s_t, a)
    q_values = policy_net(states).gather(1, actions)

    # Compute V(s_{t+1}) for all next states
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + gamma * next_q_values * (~dones)

    # Compute loss
    loss = nn.functional.mse_loss(q_values, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def visualize_dqn_policy(env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load('dqn_model.pth'))

    state, _ = env.reset()
    state = torch.FloatTensor(state).to(device)
    done = False

    plt.ion()
    while not done:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.max(0)[1].item()

        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()
        time.sleep(0.1)

        state = torch.FloatTensor(next_state).to(device)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    env = UAV3DEnv()
    train_dqn(env)
    visualize_dqn_policy(env)
