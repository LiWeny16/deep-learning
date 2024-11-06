import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from gymnasium import spaces
import pickle

class UAV3DEnv(gym.Env):
    def __init__(self, grid_size=(11, 40, 11), goal_position=(5, 39, 5), uav_radius=0.5, obstacle_radius=0.5):
        super(UAV3DEnv, self).__init__()
        self.grid_size = np.array(grid_size)
        self.uav_radius = uav_radius
        self.obstacle_radius = obstacle_radius

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
            (4, 20, 3), (6, 20, 3), (4, 20, 4), (6, 20, 4), (4, 20, 5), (6, 20, 5)
        }
        return obstacles

    def reset(self):
        self.state = np.array([5, 0, 0], dtype=np.int32)
        self.previous_states = [self.state.copy()] * 5
        self.history = [self.state.copy()]
        return self.state, {}

    def step(self, action):
        moves = np.array([
            [1, 0, 0],   # 向右
            [-1, 0, 0],  # 向左
            [0, 1, 0],   # 向前
            [0, 0, 1],   # 向上
            [0, 0, -1]   # 向下
        ])
        new_state = self.state + moves[action]

        terminated = False
        reward = -2  # 默认的步进惩罚

        # 检查是否越界
        if np.any(new_state < 0) or np.any(new_state >= self.grid_size):
            reward = -50  # 惩罚试图越界
            new_state = self.state  # 保持在当前状态
        else:
            self.state = new_state
            if tuple(self.state) in self.obstacles:
                reward = -100  # 碰撞惩罚
                terminated = True
            elif np.array_equal(self.state, self.goal_position):
                reward = 500  # 到达目标
                terminated = True

        # 更新最近5步的位置和历史位置
        self.previous_states.pop(0)
        self.previous_states.append(self.state.copy())
        self.history.append(self.state.copy())

        # 检查是否在局部位置徘徊
        if self._is_hovering():
            reward -= 30  # 给予额外的惩罚

        return self.state.copy(), reward, terminated, False, {}

    def _is_hovering(self):
        # 检查最近5步的位置是否在一个小范围内
        recent_positions = np.array(self.previous_states)
        max_distance = np.max(np.ptp(recent_positions, axis=0))
        return max_distance < 2  # 如果最大距离小于2，则认为在局部位置徘徊

    def render(self):
        plt.clf()
        ax = plt.subplot(111, projection='3d')
        self._draw_sphere(ax, self.state, self.uav_radius, 'blue')
        self._draw_sphere(ax, self.goal_position, self.uav_radius, 'green')
        for obs in self.obstacles:
            self._draw_sphere(ax, np.array(obs), self.obstacle_radius, 'red')
        # 绘制轨迹
        history = np.array(self.history)
        ax.plot(history[:, 0], history[:, 1], history[:, 2], color='blue', marker='o', markersize=2,
                linestyle='-', linewidth=1, alpha=0.6)
        # 设置轴范围
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_zlim(0, self.grid_size[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 设置图像比例
        ax.set_box_aspect(self.grid_size)
        plt.pause(0.1)

    def _draw_sphere(self, ax, center, radius, color):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=0.3)

def q_learning_train(env, episodes=3000, alpha=0.25, gamma=0.9, epsilon=1.0,
                     epsilon_min=0.1, epsilon_decay=0.995):
    q_table = np.zeros((*env.grid_size, env.action_space.n))
    success_rate = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        success = 0
        print(f"Episode: {episode + 1}/{episodes}")

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[tuple(state)])

            next_state, reward, done, _, _ = env.step(action)

            old_value = q_table[tuple(state)][action]
            next_max = np.max(q_table[tuple(next_state)])

            # Q-learning 更新
            q_table[tuple(state)][action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
            if reward == 500:
                success = 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        success_rate.append(success)

    # 绘制成功率曲线
    window = 20
    smoothed_success_rate = np.convolve(success_rate, np.ones(window) / window, mode='valid')
    plt.figure()
    plt.plot(smoothed_success_rate)
    plt.title('每集成功率')
    plt.xlabel('集数')
    plt.ylabel('成功率')
    plt.show()

    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

def q_learning_visualize(env):
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[tuple(state)])
        state, _, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.1)

if __name__ == '__main__':
    # 训练模式
    env = UAV3DEnv()
    q_learning_train(env)

    # 可视化模式
    plt.ion()
    q_learning_visualize(env)
    plt.ioff()
    plt.show()
