"""DQN智能体实现"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """深度Q网络"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 主网络和目标网络
        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.target_network = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 更新目标网络
        self.update_target_network()

    def update_target_network(self):
        """更新目标网络权重"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """根据当前状态选择动作"""
        # 探索: 随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # 利用: 选择最佳动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size: int):
        """经验回放"""
        if len(self.memory) < batch_size:
            return

        # 从记忆中随机采样一批经验
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
