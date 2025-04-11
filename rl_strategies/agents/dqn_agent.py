"""
深度Q网络(DQN)代理实现

实现了基于DQN算法的强化学习代理
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Any, Union, Optional

from rl_strategies.config import DQN_CONFIG


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化缓冲区
        
        参数:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次"""
        # 检查缓冲区是否为空
        if len(self.buffer) == 0:
            print("错误: 尝试从空缓冲区采样")
            # 返回空数组
            return (
                np.array([]), 
                np.array([], dtype=np.int64), 
                np.array([], dtype=np.float32), 
                np.array([]), 
                np.array([], dtype=np.float32)
            )
            
        # 确保批量大小不超过缓冲区大小
        batch_size = min(len(self.buffer), batch_size)
        
        # 从缓冲区随机采样
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """返回缓冲区大小"""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q网络模型"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], activation: str = 'relu'):
        """
        初始化Q网络
        
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
        
        # 创建网络层
        layers = []
        prev_dim = state_dim
        
        # 添加隐藏层
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation)
            prev_dim = dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        
        # 构建网络
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)


class DQNAgent:
    """
    深度Q网络代理
    
    基于DQN算法的强化学习代理实现
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        """
        初始化DQN代理
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置参数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or DQN_CONFIG.copy()
        
        # 提取配置参数
        self.batch_size = self.config.get('batch_size', 64)
        self.gamma = self.config.get('gamma', 0.99)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.update_target_every = self.config.get('update_target_every', 100)
        self.hidden_layers = self.config.get('hidden_layers', [128, 64])
        self.activation = self.config.get('activation', 'relu')
        self.double_dqn = self.config.get('double_dqn', False)  # 是否使用Double DQN
        
        # 性能跟踪变量，用于动态学习率调整
        self.performance_history = {
            'returns': [],          # 每回合收益率
            'rewards': [],          # 每回合奖励
            'losses': [],           # 训练损失
            'positive_actions': 0,  # 产生正面结果的行动数
            'negative_actions': 0,  # 产生负面结果的行动数
            'adaptation_cooldown': 0,  # 学习率调整冷却期
        }
        
        # 学习率动态调整参数
        self.lr_adaptation = {
            'enabled': True,               # 是否启用动态学习率
            'base_lr': self.learning_rate, # 基础学习率
            'min_lr': self.config.get('min_learning_rate', self.learning_rate / 10.0),  # 最小学习率
            'max_lr': self.config.get('max_learning_rate', None),  # 首先尝试使用max_learning_rate
        }

        # 如果max_learning_rate未设置，再检查lr_adaptation配置
        if self.lr_adaptation['max_lr'] is None and 'lr_adaptation' in self.config:
            self.lr_adaptation['max_lr'] = self.config['lr_adaptation'].get('max_lr')
        
        # 如果仍然未设置，使用默认值
        if self.lr_adaptation['max_lr'] is None:
            self.lr_adaptation['max_lr'] = self.learning_rate * 5.0
            print(f"警告: 未找到最大学习率配置，使用默认值 (基础学习率 * 5.0 = {self.lr_adaptation['max_lr']})")

        # 更新其他学习率自适应参数
        if 'lr_adaptation' in self.config:
            self.lr_adaptation.update(self.config['lr_adaptation'])

        # 添加调试信息
        print("\n" + "*"*50)
        print("[DQN初始化] 学习率配置详情:")
        print(f"[DQN初始化] 基础学习率: {self.learning_rate}")
        print(f"[DQN初始化] 配置中的max_learning_rate: {self.config.get('max_learning_rate', '未设置')}")
        print(f"[DQN初始化] 配置中的lr_adaptation: {self.config.get('lr_adaptation', '未设置')}")
        print(f"[DQN初始化] 最终使用的最大学习率: {self.lr_adaptation['max_lr']}")
        print(f"[DQN初始化] 学习率自适应启用状态: {self.lr_adaptation['enabled']}")
        print("*"*50)
        
        # 当前探索率
        self.epsilon = self.epsilon_start
        
        # 学习步数
        self.learn_step_counter = 0
        
        # 设备配置 (GPU如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建Q网络
        self.policy_net = QNetwork(
            state_dim, action_dim, self.hidden_layers, self.activation
        ).to(self.device)
        
        # 创建目标网络
        self.target_net = QNetwork(
            state_dim, action_dim, self.hidden_layers, self.activation
        ).to(self.device)
        
        # 初始化目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # 创建经验回放缓冲区
        buffer_size = self.config.get('buffer_size', 10000)
        self.memory = ReplayBuffer(buffer_size)
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
    
    def set_epsilon(self, epsilon: float) -> None:
        """
        设置探索率
        
        参数:
            epsilon: 新的探索率值
        """
        self.epsilon = max(0.0, min(1.0, epsilon))  # 确保值在[0, 1]范围内
        print(f"DQNAgent: 探索率设置为 {self.epsilon:.4f}")
    
    def replay(self, batch_size: int = None) -> Optional[float]:
        """
        从记忆库中进行经验回放

        参数:
            batch_size: 批量大小，如果为None则使用默认值
            
        返回:
            当前批次的损失值
        """
        return self.learn(force_replay=True, custom_batch_size=batch_size)
    
    def act(self, state, evaluate: bool = False, explore: bool = None) -> int:
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
            evaluate: 是否处于评估模式
            explore: 是否强制探索(兼容性参数)
            
        返回:
            选择的动作
        """
        # 兼容性处理，如果提供了explore参数
        if explore is not None:
            evaluate = not explore  # explore=True 等同于 evaluate=False
        
        # 评估模式下关闭探索
        if evaluate:
            epsilon = 0.0
        else:
            epsilon = self.epsilon
        
        # 探索: 随机选择动作，但要考虑持仓限制
        if random.random() < epsilon:
            # 获取当前持仓状态 (假设state的最后一个元素是持仓数量)
            # 如果没有持仓(position=0)，则只能选择买入(2)或持有(1)
            # 注意：这里假设position信息在状态中，具体位置需要根据实际情况调整
            has_position = False
            for i in range(len(state)):
                if abs(state[i]) > 0.01:  # 检查是否有任何非零持仓
                    has_position = True
                    break
            
            if not has_position:
                # 没有持仓时，随机选择持有(1)或买入(2)
                action = random.choice([1, 2])
            else:
                # 有持仓时，可以选择任何动作
                action = random.randrange(self.action_dim)
            
            return action
        
        # 利用: 选择Q值最大的动作，但要考虑持仓限制
        try:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:
                    state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                    state_tensor = state_tensor.to(self.device)
                
                q_values = self.policy_net(state_tensor)
                
                # 检查是否有持仓
                has_position = False
                for i in range(len(state)):
                    if abs(state[i]) > 0.01:  # 检查是否有任何非零持仓
                        has_position = True
                        break
                
                # 如果没有持仓，将卖出动作的Q值设为极小值，确保不会被选中
                if not has_position:
                    q_values[0, 0] = float('-inf')  # 0是卖出动作
                
                action = q_values.argmax(dim=1).item()
            
            return action
        except Exception as e:
            print(f"DEBUG: act方法发生异常: {e}")
            import traceback
            print(traceback.format_exc())
            # 出现异常时随机选择动作，但仍然遵守持仓规则
            has_position = False
            for i in range(len(state)):
                if abs(state[i]) > 0.01:
                    has_position = True
                    break
                    
            if not has_position:
                return random.choice([1, 2])  # 持有或买入
            else:
                return random.randrange(self.action_dim)
    
    def remember(self, state, action, reward, next_state, done) -> None:
        """
        存储经验到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.add(state, action, reward, next_state, done)
        
        # 跟踪动作结果，用于学习率调整
        if reward > 0:
            self.performance_history['positive_actions'] += 1
        elif reward < 0:
            self.performance_history['negative_actions'] += 1
    
    def learn(self, force_replay: bool = False, custom_batch_size: int = None) -> Optional[float]:
        """
        从记忆库中学习
        
        参数:
            force_replay: 是否强制进行经验回放
            custom_batch_size: 自定义批量大小
            
        返回:
            当前批次的损失值，如果没有足够样本则返回None
        """
        # 如果记忆库不够大，则不学习
        batch_size = custom_batch_size or self.batch_size
        
        # 检查记忆库是否为空或样本不足
        if len(self.memory) == 0:
            print(f"警告: 经验回放缓冲区为空，无法进行学习")
            return None
        
        if len(self.memory) < batch_size:
            if force_replay:
                print(f"警告: 经验回放缓冲区样本不足 ({len(self.memory)} < {batch_size})，使用全部可用样本")
                batch_size = len(self.memory)
            else:
                # 如果不强制回放且样本不足，则不学习
                return None
        
        # 从记忆库中采样
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 转换为Tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 使用policy_net选择动作，使用target_net评估Q值
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # 标准DQN: 使用target_net选择最大Q值
                next_q_values = self.target_net(next_states).max(1)[0]
            
            # 计算目标Q值 = 奖励 + 折扣系数 * 下一个状态的最大Q值
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = self.loss_fn(q_value, target_q_value)
        
        # 记录损失，用于性能监控
        self.performance_history['losses'].append(loss.item())
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 更新探索率
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        return loss.item()
    
    def update_performance(self, episode_return: float, episode_reward: float) -> None:
        """
        更新性能指标并调整学习率
        
        参数:
            episode_return: 回合收益率
            episode_reward: 回合累积奖励
        """
        # 记录性能
        self.performance_history['returns'].append(episode_return)
        self.performance_history['rewards'].append(episode_reward)
        
        # 打印当前学习率状态
        print(f"[DQN-学习率] 回合结束 - 当前学习率: {self.get_learning_rate():.6f}, 最小值: {self.lr_adaptation['min_lr']:.6f}, 最大值: {self.lr_adaptation['max_lr']:.6f}")
        print(f"[DQN-学习率] 学习率参数: 启用={self.lr_adaptation['enabled']}, 收益率={episode_return:.2f}%")
        
        # 如果动态学习率功能被禁用，则直接返回
        if not self.lr_adaptation['enabled']:
            print(f"[DQN-学习率] 动态学习率已禁用，保持当前学习率 {self.get_learning_rate():.6f}")
            return
        
        # 如果冷却期未结束，减少计数器
        if self.performance_history['adaptation_cooldown'] > 0:
            self.performance_history['adaptation_cooldown'] -= 1
            print(f"[DQN-学习率] 学习率调整冷却中，剩余冷却期 {self.performance_history['adaptation_cooldown']} 回合")
            return
        
        # 检查是否有足够的数据做出决策
        if len(self.performance_history['returns']) < self.lr_adaptation['adaptation_window']:
            print(f"[DQN-学习率] 收益数据不足，需要至少 {self.lr_adaptation['adaptation_window']} 回合数据，当前 {len(self.performance_history['returns'])} 回合")
            
            # 尝试使用简化逻辑，确保至少有一些调整发生
            print(f"[DQN-学习率] 使用简化逻辑进行调整，基于当前回合收益 {episode_return:.2f}%")
            old_lr = self.get_learning_rate()
            
            # 根据收益正负决定增减学习率
            if episode_return < 0:
                # 负收益，增加学习率
                new_lr = min(old_lr * 1.1, self.lr_adaptation['max_lr'])
                adjustment_type = "增加"
                adjustment_reason = f"负收益 ({episode_return:.2f}%)"
            else:
                # 正收益，减少学习率
                new_lr = max(old_lr * 0.9, self.lr_adaptation['min_lr'])
                adjustment_type = "减少"
                adjustment_reason = f"正收益 ({episode_return:.2f}%)"
                
            # 应用新的学习率
            if abs(new_lr - old_lr) > 1e-6:  # 确保有明显变化
                self.set_learning_rate(new_lr)
                print(f"[DQN-学习率] 简化调整 - {adjustment_type}学习率从 {old_lr:.6f} 到 {new_lr:.6f}，原因: {adjustment_reason}")
                return True
            
            return False
        
        # 只检查最近的几个回合
        recent_returns = self.performance_history['returns'][-self.lr_adaptation['adaptation_window']:]
        avg_return = sum(recent_returns) / len(recent_returns)
        
        # 计算波动性（收益的标准差）
        if len(recent_returns) > 1:
            return_std = np.std(recent_returns)
        else:
            return_std = 0
        
        # 检查正负动作比例
        total_actions = self.performance_history['positive_actions'] + self.performance_history['negative_actions']
        if total_actions > 0:
            positive_ratio = self.performance_history['positive_actions'] / total_actions
        else:
            positive_ratio = 0.5  # 默认值
            
        print(f"[DQN-学习率] 分析数据: 平均收益={avg_return:.2f}%, 收益波动={return_std:.2f}, 正向比例={positive_ratio:.2f}")
        
        # 重置动作计数器
        self.performance_history['positive_actions'] = 0
        self.performance_history['negative_actions'] = 0
        
        # 学习率调整决策
        old_lr = self.get_learning_rate()
        
        # 如果收益持续为负且低于阈值，增加学习率以促进探索
        if avg_return < self.lr_adaptation['increase_threshold'] or positive_ratio < 0.3:
            new_lr = min(old_lr * self.lr_adaptation['increase_factor'], self.lr_adaptation['max_lr'])
            adjustment_type = "增加"
            adjustment_reason = "表现不佳"
            if avg_return < self.lr_adaptation['increase_threshold']:
                adjustment_reason += f"，平均收益率({avg_return:.2f})低于阈值({self.lr_adaptation['increase_threshold']:.2f})"
            if positive_ratio < 0.3:
                adjustment_reason += f"，正向动作比例({positive_ratio:.2f})过低"
        
        # 如果收益良好且高于阈值，降低学习率以细化策略
        elif avg_return > self.lr_adaptation['decrease_threshold'] and positive_ratio > 0.6:
            new_lr = max(old_lr * self.lr_adaptation['decrease_factor'], self.lr_adaptation['min_lr'])
            adjustment_type = "减少"
            adjustment_reason = f"表现良好，平均收益率({avg_return:.2f})高于阈值({self.lr_adaptation['decrease_threshold']:.2f})"
        
        # 收益波动过大，适当减小学习率以提高稳定性
        elif return_std > 0.2 and avg_return > 0:
            new_lr = max(old_lr * 0.9, self.lr_adaptation['min_lr'])
            adjustment_type = "减小"
            adjustment_reason = f"波动较大(σ={return_std:.2f})但平均收益为正"
        
        # 不需要调整
        else:
            print(f"[DQN-学习率] 当前表现合适，保持学习率 {old_lr:.6f}，平均收益率={avg_return:.2f}，正向动作比例={positive_ratio:.2f}")
            return False
        
        # 应用新的学习率
        if abs(new_lr - old_lr) > 1e-6:  # 确保有明显变化
            self.set_learning_rate(new_lr)
            print(f"[DQN-学习率] 动态学习率调整 - {adjustment_type}学习率从 {old_lr:.6f} 到 {new_lr:.6f}，原因: {adjustment_reason}")
            
            # 设置冷却期，避免频繁调整
            self.performance_history['adaptation_cooldown'] = self.lr_adaptation['cooldown_period']
            return True  # 返回True表示学习率已调整
        
        return False  # 返回False表示学习率未调整
    
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float) -> None:
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存策略网络
        torch.save(self.policy_net.state_dict(), os.path.join(path, "policy_net.pth"))
        
        # 保存目标网络
        torch.save(self.target_net.state_dict(), os.path.join(path, "target_net.pth"))
        
        # 保存优化器状态
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        
        # 保存代理状态
        agent_state = {
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter,
            'config': self.config,
            'performance_history': self.performance_history,
            'lr_adaptation': self.lr_adaptation
        }
        torch.save(agent_state, os.path.join(path, "agent_state.pth"))
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        # 加载策略网络
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, "policy_net.pth"), map_location=self.device)
        )
        
        # 加载目标网络
        self.target_net.load_state_dict(
            torch.load(os.path.join(path, "target_net.pth"), map_location=self.device)
        )
        
        # 加载优化器状态
        self.optimizer.load_state_dict(
            torch.load(os.path.join(path, "optimizer.pth"), map_location=self.device)
        )
        
        # 加载代理状态
        state = torch.load(os.path.join(path, "agent_state.pth"), map_location=self.device)
        self.epsilon = state['epsilon']
        self.learn_step_counter = state['learn_step_counter']
        self.config = state['config']
        
        # 加载性能历史(如果有)
        if 'performance_history' in state:
            self.performance_history = state['performance_history']
        
        # 加载学习率适应参数(如果有)
        if 'lr_adaptation' in state:
            self.lr_adaptation = state['lr_adaptation']
    
    def get_weights(self):
        """
        获取模型权重
        
        返回:
            当前策略网络的权重字典
        """
        return self.policy_net.state_dict()
    
    def set_weights(self, weights):
        """
        设置模型权重
        
        参数:
            weights: 权重字典或状态字典
        """
        try:
            # 如果是完整的权重字典（包含policy_net、target_net等）
            if isinstance(weights, dict) and 'policy_net' in weights:
                self.policy_net.load_state_dict(weights['policy_net'])
                self.target_net.load_state_dict(weights['target_net'])
                self.optimizer.load_state_dict(weights['optimizer'])
                self.epsilon = weights['epsilon']
                self.learn_step_counter = weights['learn_step_counter']
                
                # 更新性能历史和学习率适应参数（如果存在）
                if 'performance_history' in weights:
                    self.performance_history = weights['performance_history']
                if 'lr_adaptation' in weights:
                    self.lr_adaptation = weights['lr_adaptation']
            # 如果只是策略网络的状态字典
            else:
                self.policy_net.load_state_dict(weights)
                self.target_net.load_state_dict(weights)
                
            # 设置目标网络为评估模式
            self.target_net.eval()
            
        except Exception as e:
            print(f"设置模型权重时出错: {e}")
            import traceback
            print(traceback.format_exc())