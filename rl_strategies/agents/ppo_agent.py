"""
近端策略优化(PPO)代理实现

实现了基于PPO算法的强化学习代理，适用于连续动作空间
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Any, Union, Optional, Callable

from rl_strategies.config import PPO_CONFIG


class ActorCritic(nn.Module):
    """PPO的Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], activation: str = 'tanh'):
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
            self.activation = nn.Tanh()
        
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
    
    def evaluate(self, state, action):
        """
        评估给定状态和动作
        
        参数:
            state: 状态批次
            action: 动作批次
            
        返回:
            动作概率分布, 动作对数概率, 状态价值, 熵
        """
        action_probs, state_value = self.forward(state)
        
        # 创建分类分布
        dist = Categorical(action_probs)
        
        # 计算选择动作的对数概率
        action_log_probs = dist.log_prob(action)
        
        # 计算分布的熵
        dist_entropy = dist.entropy()
        
        return action_probs, action_log_probs, state_value, dist_entropy


class PPOMemory:
    """PPO算法的经验回放缓冲区"""
    
    def __init__(self):
        """初始化缓冲区"""
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, prob, reward, value, done):
        """添加经验到缓冲区"""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_batch(self):
        """获取整个批次数据"""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones)
        )


class PPOAgent:
    """
    近端策略优化(PPO)代理
    
    基于PPO算法的强化学习代理实现
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        """
        初始化PPO代理
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置参数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PPO_CONFIG.copy()
        
        # 提取配置参数
        self.hidden_layers = self.config.get('hidden_layers', [128, 64])
        self.activation = self.config.get('activation', 'tanh')
        self.learning_rate = self.config.get('learning_rate', 0.0003)
        self.gamma = self.config.get('gamma', 0.99)
        self.lambda_gae = self.config.get('lambda_gae', 0.95)
        self.clip_ratio = self.config.get('clip_ratio', 0.2)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.epochs_per_update = self.config.get('epochs_per_update', 4)
        self.batch_size = self.config.get('batch_size', 64)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建Actor-Critic网络
        self.policy = ActorCritic(
            state_dim, action_dim, self.hidden_layers, self.activation
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # 创建经验回放缓冲区
        self.memory = PPOMemory()
        
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
            action, action_prob = self.policy.act(state_tensor, deterministic=evaluate)
        
        return action
    
    def remember(self, state, action, reward, next_state, done) -> None:
        """
        存储经验到缓冲区
        
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
            action_prob = action_probs[0][action].item()
            value = state_value.item()
        
        # 存储到内存
        self.memory.add(state, action, action_prob, reward, value, done)
    
    def learn(self) -> Optional[float]:
        """
        从经验中学习
        
        返回:
            当前Actor损失值(如果学习发生)
        """
        # 获取缓冲区数据
        states, actions, old_probs, rewards, values, dones = self.memory.get_batch()
        
        # 如果数据不足，则不学习
        if len(states) < self.batch_size:
            self.memory.clear()
            return None
        
        # 计算GAE和目标值
        advantages, returns = self._compute_advantages_and_returns(rewards, values, dones)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_probs_tensor = torch.FloatTensor(old_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # 学习步数+1
        self.learn_step_counter += 1
        
        # 多次训练，提高样本利用率
        actor_losses = []
        
        for _ in range(self.epochs_per_update):
            # 打乱数据索引
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # 分批次训练
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # 获取批次数据
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_probs = old_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # 评估当前策略
                _, log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # 计算比率
                ratios = torch.exp(log_probs - torch.log(batch_old_probs))
                
                # 计算裁剪后的损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 记录Actor损失
                actor_losses.append(actor_loss.item())
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # 清空缓冲区
        self.memory.clear()
        
        # 返回平均Actor损失
        return sum(actor_losses) / len(actor_losses) if actor_losses else None
    
    def _compute_advantages_and_returns(self, rewards, values, dones):
        """
        计算广义优势估计(GAE)和回报
        
        参数:
            rewards: 奖励序列
            values: 值函数预测
            dones: 结束标志
            
        返回:
            优势和回报
        """
        # 计算优势估计
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # 下一个状态的值函数估计(用于最后一个时间步)
        next_value = 0
        gae = 0
        
        # 反向计算GAE
        for t in reversed(range(len(rewards))):
            # 计算时序差分目标
            if t == len(rewards) - 1:
                # 对于最后一步，使用0作为下一个值(或者可以用下一个状态的值函数预测)
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            
            # 更新GAE
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            advantages[t] = gae
        
        # 计算回报(值函数训练的目标)
        returns = advantages + values
        
        return advantages, returns
    
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