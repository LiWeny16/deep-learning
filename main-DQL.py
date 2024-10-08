import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from gymnasium import spaces

from collections import deque
import matplotlib.pyplot as plt
import time

class UAV3DEnv(gym.Env):
    def __init__(self, grid_size=(11, 11, 11), goal_position=(10, 10, 10), uav_radius=0.5, obstacle_radius=0.5):
        super().__init__()
        self.grid_size = grid_size
        self.uav_radius = uav_radius
        self.obstacle_radius = obstacle_radius
        
        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(3,), dtype=np.int32)
        self.action_space = spaces.Discrete(6)

        self.state = np.array([0, 0, 0], dtype=np.int32)
        self.goal_position = np.array(goal_position, dtype=np.int32)
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < 20:
            obstacle = tuple(np.random.randint(0, max(self.grid_size), size=3))
            if obstacle != tuple(self.state) and obstacle != tuple(self.goal_position):
                obstacles.add(obstacle)
        return list(obstacles)
    
    def reset(self):
        self.state = np.array([0, 0, 0], dtype=np.int32)
        self.obstacles = self._generate_obstacles()  # 重置障碍物
        return self.state, {}

    def step(self, action):
        # 定义动作：6个方向 (X±1, Y±1, Z±1)
        moves = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        self.state = np.clip(self.state + moves[action], 0, np.array(self.grid_size) - 1)
        
        reward = -1
        terminated = False
        if tuple(self.state) in self.obstacles:
            reward = -10
            terminated = True
        elif np.array_equal(self.state, self.goal_position):
            reward = 100
            terminated = True
        
        return self.state, reward, terminated, False, {}

    def render(self):
        plt.clf()
        ax = plt.subplot(111, projection='3d')

        # 绘制UAV,目标和障碍物
        self._draw_sphere(ax, self.state, self.uav_radius, 'blue', 'UAV')
        self._draw_sphere(ax, self.goal_position, self.uav_radius, 'green', 'Goal')
        for obs in self.obstacles:
            self._draw_sphere(ax, obs, self.obstacle_radius, 'red')

        ax.set(xlim=(0, self.grid_size[0]), ylim=(0, self.grid_size[1]), zlim=(0, self.grid_size[2]),
               xlabel='X', ylabel='Y', zlabel='Z')
        plt.pause(0.1)

    def _draw_sphere(self, ax, center, radius, color, label=None):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=0.3, label=label)


# 定义DQN的神经网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 创建策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算当前Q值
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # 计算损失并优化
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_dqn(env, agent, episodes=500):
    success_rate = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        success = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            # 将经验存储到回放缓冲区
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # 更新网络
            agent.update()
            if np.array_equal(state, env.goal_position) and reward == 100:
                success = 1

        # 每隔一定的间隔更新目标网络
        if episode % 10 == 0:
            agent.update_target_net()

        agent.decay_epsilon()
        success_rate.append(success)

    # 绘制成功率
    plt.plot(np.convolve(success_rate, np.ones(20)/20, mode='valid'))
    plt.title('Success Rate per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.show()

def dqn_visualize(env, agent):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        state, _, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.1)

# 创建UAV环境
env = UAV3DEnv()

# 创建DQN智能体
state_dim = np.prod(env.grid_size)
action_dim = env.action_space.n
agent = DQNAgent(state_dim=3, action_dim=action_dim)

# 训练DQN
train_dqn(env, agent)

# 可视化DQN执行
plt.ion()
dqn_visualize(env, agent)
plt.ioff()
plt.show()
