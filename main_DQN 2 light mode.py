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

# 定义UAV 3D环境
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
            # 定义障碍物的坐标集合
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
            [1, 0, 0],    # 右
            [-1, 0, 0],   # 左
            [0, 1, 0],    # 前
            [0, 0, 1],    # 上
            [0, 0, -1]    # 下
        ])
        new_state = self.state + moves[action]
        terminated = False
        reward = -1  # 每步的基本奖励

        # 检查是否越界
        if np.any(new_state < 0) or np.any(new_state >= self.grid_size):
            reward = -10  # 碰壁惩罚
            new_state = self.state.copy()
        else:
            self.state = new_state.copy()
            if tuple(self.state) in self.obstacles:
                reward = -100  # 碰到障碍物的惩罚
                terminated = True
            elif np.array_equal(self.state, self.goal_position):
                reward = 100  # 到达目标的奖励
                terminated = True

        # 更新之前的状态和历史记录
        self.previous_states.pop(0)
        self.previous_states.append(self.state.copy())
        self.history.append(self.state.copy())

        # 额外的悬停惩罚
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
        # 绘制轨迹
        history = np.array(self.history)
        ax.plot(history[:, 0], history[:, 1], history[:, 2], color='blue', marker='o', markersize=2,
                linestyle='-', linewidth=1, alpha=0.6)
        # 设置坐标轴范围
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

# 定义经验回放缓冲区
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

# 定义带有双重DQN的Dueling DQN网络
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

# 使用双重DQN的深度Q学习算法
def train_dqn(env, num_episodes=1000, batch_size=64, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=1000,
              target_update=10, memory_capacity=10000):
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

            # Epsilon-Greedy动作选择
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
            reward_tensor = torch.FloatTensor([reward]).to(device)
            done_tensor = torch.FloatTensor([done]).to(device)

            memory.push(state.cpu().numpy(), action, reward, next_state, done)

            state = next_state_tensor

            # 进行优化步骤
            if len(memory) >= batch_size:
                optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device)

        # 更新目标网络
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # 绘制训练进程
    plt.figure()
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # 保存训练好的模型
    torch.save(policy_net.state_dict(), 'dqn_model.pth')

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = torch.FloatTensor(states).squeeze().to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).squeeze().to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # 计算Q(s_t, a)
    q_values = policy_net(states).gather(1, actions)

    # 计算期望的Q值
    with torch.no_grad():
        # 双重DQN：使用策略网络选择动作，目标网络评估
        next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = target_net(next_states).gather(1, next_actions)
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    # 计算损失
    loss = nn.functional.mse_loss(q_values, expected_q_values)

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def visualize_dqn_policy(env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load('dqn_model.pth'))
    policy_net.eval()

    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    done = False

    plt.ion()
    while not done:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()

        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()
        time.sleep(0.1)

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    env = UAV3DEnv()

    # 添加 prefer_mode 变量，可以设置为 'train' 或 'demo'
    prefer_mode = 'train'  # 修改此变量以选择运行模式

    if prefer_mode == 'train':
        print('训练模式：开始训练模型。')
        train_dqn(env)
        print('训练完成，开始可视化。')
        visualize_dqn_policy(env)
    elif prefer_mode == 'demo':
        if os.path.exists('dqn_model.pth'):
            print('演示模式：加载已有模型进行可视化。')
            visualize_dqn_policy(env)
        else:
            print('未找到模型，请先在训练模式下训练模型。')
    else:
        print("无效的 prefer_mode 值，请将其设置为 'train' 或 'demo'。")
