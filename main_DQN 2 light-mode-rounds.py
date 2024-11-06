import os
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

# Define the UAV 3D environment
class UAV3DEnv(gym.Env):
    def __init__(self, grid_size=(11, 40, 11), goal_position=(5, 39, 5), max_steps=450):
        super(UAV3DEnv, self).__init__()
        self.grid_size = np.array(grid_size)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32)
        self.action_space = spaces.Discrete(5)
        self.goal_position = np.array(goal_position, dtype=np.int32)
        self.obstacles = self._generate_obstacles()
        self.max_steps = max_steps
        self.reset()

    def _generate_obstacles(self):
        obstacles = {
            # Define the set of obstacle coordinates
            (2, 10, 0), (2, 11, 0), (2, 12, 0), (2, 10, 1), (2, 11, 1), (2, 12, 1),
            (2, 10, 2), (2, 11, 2), (2, 12, 2), (2, 10, 3), (2, 11, 3), (2, 12, 3),
            (8, 10, 0), (8, 11, 0), (8, 12, 0), (8, 10, 1), (8, 11, 1), (8, 12, 1),
            (8, 10, 2), (8, 11, 2), (8, 12, 2), (8, 10, 3), (8, 11, 3), (8, 12, 3),
            (4, 30, 0), (5, 30, 0), (6, 30, 0), (7, 30, 0), (4, 30, 1), (5, 30, 1),
            (6, 30, 1), (7, 30, 1), (4, 30, 2), (5, 30, 2), (6, 30, 2), (7, 30, 2),
            (4, 20, 0), (6, 20, 0), (4, 20, 1), (6, 20, 1), (4, 20, 2), (6, 20, 2),
            (4, 20, 3), (6, 20, 3), (4, 20, 4), (6, 20, 4), (4, 20, 5), (6, 20, 5),
        }
        return obstacles

    def reset(self):
        self.steps = 0
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
        reward = -1  # Basic reward for each step

        # Check for boundary conditions
        if np.any(new_state < 0) or np.any(new_state >= self.grid_size):
            reward = -10  # Penalty for hitting walls
            new_state = self.state.copy()
        else:
            self.state = new_state.copy()
            if tuple(self.state) in self.obstacles:
                reward = -100  # Penalty for hitting obstacles
                terminated = True
            elif np.array_equal(self.state, self.goal_position):
                reward = 100  # Reward for reaching the goal
                terminated = True

        # Update previous states and history
        self.previous_states.pop(0)
        self.previous_states.append(self.state.copy())
        self.history.append(self.state.copy())

        # Additional penalty for hovering
        if self._is_hovering():
            reward -= 5

        self.steps += 1
        if self.steps >= self.max_steps:
            terminated = True

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
        # Draw the trajectory
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

# Define the replay buffer
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

# Define the Dueling DQN network with Double DQN
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
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Deep Q-Learning algorithm with Double DQN
def train_dqn(env, num_episodes=1000, batch_size=64, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=1000,
              target_update=10, memory_capacity=10000, model_save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayBuffer(memory_capacity)

    steps_done = 0
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        done = False

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1

            # Epsilon-Greedy action selection
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.max(1)[1].item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            memory.push(state.cpu().numpy(), action, reward, next_state, done)

            state = next_state_tensor

            # Optimize the model
            if len(memory) >= batch_size:
                optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device)

        # Update the target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Save the trained model
    if model_save_path:
        torch.save(policy_net.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = torch.FloatTensor(states).squeeze().to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).squeeze().to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Compute Q(s_t, a)
    q_values = policy_net(states).gather(1, actions)

    # Compute the expected Q values
    with torch.no_grad():
        # Double DQN: use policy network to select actions, target network to evaluate
        next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = target_net(next_states).gather(1, next_actions)
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss
    loss = nn.functional.mse_loss(q_values, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate_policy(env, policy_net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    done = False

    while not done:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    # Check if the episode was successful
    success = reward == 100
    return success

def visualize_cumulative_success_rate(success_list):
    cumulative_success = np.cumsum(success_list)
    rounds = np.arange(1, len(success_list) + 1)
    cumulative_success_rate = cumulative_success / rounds * 100

    plt.figure()
    plt.plot(rounds, cumulative_success_rate, marker='o')
    plt.title('Cumulative Success Rate Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Success Rate (%)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    env = UAV3DEnv()

    # Set 'prefer_mode' variable to 'train' or 'demo'
    prefer_mode = 'demo'  # Change this variable to select the mode
    n_rounds = 5  # Number of training rounds (n >= 1)

    if prefer_mode == 'train':
        print('Training mode: Starting training.')
        if not os.path.exists('model'):
            os.makedirs('model')

        for round_num in range(1, n_rounds + 1):
            print(f"\nStarting training round {round_num}/{n_rounds}")
            model_save_path = f'model/round_{round_num}_dqn_model.pth'
            train_dqn(env, model_save_path=model_save_path)
            print(f"Training round {round_num} completed.")

    elif prefer_mode == 'demo':
        print('Demo mode: Loading models and evaluating policies.')
        model_folder = 'model'
        success_list = []
        cumulative_success_rates = []
        model_files = sorted([f for f in os.listdir(model_folder) if f.startswith('round_') and f.endswith('.pth')])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        for idx, model_file in enumerate(model_files, 1):
            model_path = os.path.join(model_folder, model_file)
            print(f"\nEvaluating model: {model_file}")

            policy_net = DuelingDQN(state_dim, action_dim).to(device)
            policy_net.load_state_dict(torch.load(model_path))
            policy_net.eval()

            # Run a single episode and record success or failure
            success = evaluate_policy(env, policy_net)
            success_list.append(int(success))
            cumulative_success = np.cumsum(success_list)
            cumulative_success_rate = cumulative_success[-1] / idx * 100
            cumulative_success_rates.append(cumulative_success_rate)

            result = 'Success' if success else 'Failure'
            print(f"Round {idx}: {result}, Cumulative Success Rate: {cumulative_success_rate:.2f}%")

        # Plot the cumulative success rate
        visualize_cumulative_success_rate(success_list)

    else:
        print("Invalid 'prefer_mode' value. Please set it to 'train' or 'demo'.")