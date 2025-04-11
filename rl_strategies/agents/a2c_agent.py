"""
优势演员-批评家(A2C)代理实现

实现了基于A2C算法的强化学习代理
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Any, Union, Optional

from rl_strategies.config import A2C_CONFIG


class ActorCritic(nn.Module):
    """A2C的Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], activation: str = 'relu'):
        """
        初始化Actor-Critic网络
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_layers: 隐藏层神经元数量列表
            activation: 激活函数
        """
        super().__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # 创建共享特征提取网络
        self.feature_layers = []
        prev_dim = state_dim
        
        for dim in hidden_layers[:-1]:  # 除了最后一层
            self.feature_layers.append(nn.Linear(prev_dim, dim))
            self.feature_layers.append(self.activation)
            prev_dim = dim
        
        self.feature_network = nn.Sequential(*self.feature_layers)
        
        # Actor网络(策略)
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            self.activation,
            nn.Linear(hidden_layers[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络(价值)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            self.activation,
            nn.Linear(hidden_layers[-1], 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入状态
            
        返回:
            动作概率分布和状态价值
        """
        features = self.feature_network(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def act(self, state, deterministic=False):
        """
        根据状态选择动作
        
        参数:
            state: 当前状态
            deterministic: 是否使用确定性策略(选择概率最高的动作)
            
        返回:
            选择的动作和动作概率
        """
        action_probs, _ = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
            return action, action_probs[0][action].item()
        else:
            # 构建分类分布并采样
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
            # 返回动作和对应概率
            return action, action_probs[0][action].item()


class A2CAgent:
    """
    优势演员-批评家(A2C)代理
    
    基于A2C算法的强化学习代理实现
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        """
        初始化A2C代理
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置参数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or A2C_CONFIG.copy()
        
        # 提取配置参数
        self.hidden_layers = self.config.get('hidden_layers', [128, 64])
        self.activation = self.config.get('activation', 'relu')
        self.learning_rate = self.config.get('learning_rate', 0.0005)
        self.gamma = self.config.get('gamma', 0.99)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建Actor-Critic网络
        self.policy = ActorCritic(
            state_dim, action_dim, self.hidden_layers, self.activation
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # 存储状态、动作、奖励等
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # 学习步数
        self.learn_step_counter = 0
    
    def act(self, state, evaluate: bool = False) -> int:
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
            evaluate: 是否处于评估模式(确定性)
            
        返回:
            选择的动作
        """
        # 转换状态为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 选择动作
            action, _ = self.policy.act(state_tensor, deterministic=evaluate)
        
        return action
    
    def remember(self, state, action, reward, next_state, done) -> None:
        """
        存储经验
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 转换状态为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取动作概率和状态价值
            action_probs, state_value = self.policy(state_tensor)
            
            # 创建分类分布
            dist = Categorical(action_probs)
            
            # 计算对数概率
            log_prob = dist.log_prob(torch.tensor([action]).to(self.device)).item()
        
        # 存储经验
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(state_value.item())
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def learn(self) -> Optional[float]:
        """
        从经验中学习
        
        返回:
            当前Actor损失值(如果学习发生)
        """
        # 如果没有足够的经验，则不学习
        if len(self.states) <= 1:
            return None
        
        # 转换为numpy数组
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)
        
        # 计算优势和回报
        returns = []
        advantages = []
        R = 0
        
        # 反向计算回报和优势
        for i in reversed(range(len(rewards))):
            # 计算折扣回报
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
            
            # 简单的优势估计: R - V(s)
            advantages.insert(0, R - values[i])
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        
        # 正向传播
        action_probs, state_values = self.policy(states_tensor)
        
        # 计算新的动作分布
        dist = Categorical(action_probs)
        
        # 计算新的动作对数概率
        new_log_probs = dist.log_prob(actions_tensor)
        
        # 计算熵
        entropy = dist.entropy().mean()
        
        # 计算Actor损失
        actor_loss = -(new_log_probs * advantages_tensor).mean()
        
        # 计算Critic损失
        critic_loss = nn.MSELoss()(state_values.squeeze(-1), returns_tensor)
        
        # 总损失
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # 学习步数+1
        self.learn_step_counter += 1
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return actor_loss.item()
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存策略网络
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pth"))
        
        # 保存优化器状态
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        
        # 保存配置
        torch.save({
            'learn_step_counter': self.learn_step_counter,
            'config': self.config
        }, os.path.join(path, "agent_state.pth"))
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        # 加载策略网络
        self.policy.load_state_dict(
            torch.load(os.path.join(path, "policy.pth"), map_location=self.device)
        )
        
        # 加载优化器状态
        self.optimizer.load_state_dict(
            torch.load(os.path.join(path, "optimizer.pth"), map_location=self.device)
        )
        
        # 加载代理状态
        state = torch.load(os.path.join(path, "agent_state.pth"), map_location=self.device)
        self.learn_step_counter = state['learn_step_counter']
        self.config = state['config'] 