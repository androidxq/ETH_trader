"""
神经网络模型实现

包含各种神经网络架构，用于强化学习代理
"""

import torch
import torch.nn as nn
from typing import List, Union, Callable, Optional, Tuple


class MLPNetwork(nn.Module):
    """
    多层感知机网络
    
    适用于RL的灵活MLP网络结构，支持各种配置和激活函数
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: Union[str, Callable] = 'relu',
        output_activation: Optional[Union[str, Callable]] = None,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        初始化MLP网络
        
        参数:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_layers: 隐藏层维度列表
            activation: 激活函数
            output_activation: 输出层激活函数
            use_batch_norm: 是否使用批归一化
            dropout_rate: Dropout比率
        """
        super().__init__()
        
        # 选择激活函数
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation == 'elu':
                self.activation = nn.ELU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            else:
                self.activation = nn.ReLU()
        else:
            self.activation = activation
        
        # 选择输出激活函数
        if isinstance(output_activation, str):
            if output_activation == 'relu':
                self.output_activation = nn.ReLU()
            elif output_activation == 'tanh':
                self.output_activation = nn.Tanh()
            elif output_activation == 'sigmoid':
                self.output_activation = nn.Sigmoid()
            elif output_activation == 'softmax':
                self.output_activation = nn.Softmax(dim=-1)
            else:
                self.output_activation = None
        else:
            self.output_activation = output_activation
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
                
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 如果有输出激活函数，添加
        if self.output_activation is not None:
            layers.append(self.output_activation)
        
        # 构建网络
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """
    双重Q网络
    
    用于实现Dueling DQN架构，把值函数分为状态值和优势函数
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int],
        activation: Union[str, Callable] = 'relu'
    ):
        """
        初始化双重Q网络
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_layers: 隐藏层维度列表
            activation: 激活函数
        """
        super().__init__()
        
        # 选择激活函数
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            else:
                self.activation = nn.ReLU()
        else:
            self.activation = activation
        
        # 构建特征提取网络
        self.feature_network = MLPNetwork(
            input_dim=state_dim,
            output_dim=hidden_layers[-1],
            hidden_layers=hidden_layers[:-1],
            activation=self.activation
        )
        
        # 构建值函数网络
        self.value_network = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1]//2),
            self.activation,
            nn.Linear(hidden_layers[-1]//2, 1)
        )
        
        # 构建优势函数网络
        self.advantage_network = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1]//2),
            self.activation,
            nn.Linear(hidden_layers[-1]//2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            Q值张量
        """
        features = self.feature_network(x)
        
        # 计算状态值
        value = self.value_network(features)
        
        # 计算优势函数
        advantages = self.advantage_network(features)
        
        # 合并状态值和优势函数: Q = V + (A - mean(A))
        # 这里使用优势函数减去均值以保持Q值的估计
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class NoisyLinear(nn.Module):
    """
    噪声线性层
    
    用于实现NoisyNet，增加探索性
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        """
        初始化噪声线性层
        
        参数:
            in_features: 输入特征数
            out_features: 输出特征数
            std_init: 噪声标准差的初始化值
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        # 初始化参数
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """重置参数"""
        mu_range = 1 / self.in_features ** 0.5
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features ** 0.5)
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_features ** 0.5)
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # 外积生成噪声矩阵
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """
        缩放噪声
        
        参数:
            size: 噪声维度
            
        返回:
            缩放后的噪声
        """
        noise = torch.randn(size)
        return noise.sign().mul(noise.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return nn.functional.linear(x, weight, bias) 