import random
from collections import deque

import numpy as np
import torch
from torch import nn, optim


class DQNAgent:
    def __init__(self, state_size, max_actions, device="cpu"):
        self.state_size = state_size
        self.max_actions = max_actions  # 最大可能动作数
        self.device = device

        # 超参数
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=20000)

        # 网络
        self.policy_net = self._build_network().to(device)
        self.target_net = self._build_network().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_freq = 100
        self.steps_done = 0

    def _build_network(self):
        """构建神经网络"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.max_actions),
        )

    def select_action(self, state, legal_actions):
        """
        选择动作
        只从合法动作中选择
        """
        if not legal_actions:
            return None  # 没有合法动作

        if random.random() < self.epsilon:
            # 探索：随机选择合法动作
            return random.choice(legal_actions)
        else:
            # 利用：选择Q值最高的合法动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)[0]

                # 只考虑合法动作
                legal_q_values = {action: q_values[action].item() for action in legal_actions}

                # 选择Q值最高的动作
                return max(legal_q_values, key=legal_q_values.get)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """从经验中学习"""
        if len(self.memory) < self.batch_size:
            return 0

        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        # 目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失
        loss = self.criterion(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
