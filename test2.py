import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 不要忘记导入这个模块

class GridWorld3D:
    def __init__(self, grid_size, goal_position, obstacles):
        self.grid_size = grid_size
        self.goal_position = goal_position
        self.obstacles = obstacles
        self.state = [0, 0, 0]  # 初始位置

    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制无人机
        ax.scatter(*self.state, color='blue', label='Drone')
        
        # 绘制目标
        ax.scatter(*self.goal_position, color='green', label='Goal')
        
        # 绘制障碍物
        for obs in self.obstacles:
            ax.scatter(*obs, color='red', label='Obstacle' if obs == self.obstacles[0] else "")
        
        # 设置网格大小
        ax.set_xlim(0, self.grid_size[0] - 1)
        ax.set_ylim(0, self.grid_size[1] - 1)
        ax.set_zlim(0, self.grid_size[2] - 1)
        
        # 添加标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.legend()
        plt.show()

# 示例使用
grid_size = (4, 4, 4)
goal_position = (3, 3, 3)
obstacles = [(1, 1, 1), (2, 2, 2), (1, 2, 3)]

env = GridWorld3D(grid_size, goal_position, obstacles)
env.render()