"""华容道强化学习训练脚本"""

import os
import sys
from collections import deque

import numpy as np
import torch

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dqn_agent import DQNAgent
from src.rl_env import HuarongdaoEnv


def train(episodes: int = 1000):
    """训练函数"""
    # 创建环境和智能体
    env = HuarongdaoEnv()
    state_size = 4 * 5  # 4行5列的棋盘
    # 动作空间: 10个棋子 * 4个方向 = 40个动作
    action_size = 40
    agent = DQNAgent(state_size, action_size)

    # 训练日志
    scores = deque(maxlen=100)

    for episode in range(episodes):
        state = env.reset()
        state = state.flatten()
        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action = agent.act(state)

            # 执行动作
            # 将动作索引转换为实际的动作 (piece_id, dx, dy)
            piece_id = action // 4 + 1  # 棋子ID (1-10)
            direction = action % 4  # 方向 (0-3)

            # 将方向映射为(dx, dy)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dx, dy = directions[direction]

            next_state, reward, done = env.step((piece_id, dx, dy))
            next_state = next_state.flatten()

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            total_reward += reward

            # 经验回放
            if len(agent.memory) > 32:
                agent.replay(32)

        # 记录分数
        scores.append(total_reward)

        # 更新目标网络
        if episode % 100 == 0:
            agent.update_target_network()

        # 打印进度
        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

    # 保存模型
    torch.save(agent.q_network.state_dict(), "huarongdao_dqn.pth")
    print("Model saved!")


if __name__ == "__main__":
    train(1000)
