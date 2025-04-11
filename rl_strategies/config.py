"""
强化学习策略配置文件

包含所有强化学习训练和环境配置参数
"""

# 默认环境配置
DEFAULT_ENV_CONFIG = {
    # 基础环境设置
    "window_size": 20,           # 观察窗口大小
    "max_steps": 500,            # 每个回合最大步数
    "initial_balance": 10000,    # 初始资金
    "transaction_fee": 0.0005,   # 交易手续费率（0.05%）
    
    # 仓位和风险控制
    "fixed_trade_amount": 100.0, # 固定每次交易金额为100
    "max_position_size": 0.7,    # 最大仓位比例
    "base_position_size": 0.3,   # 基础仓位比例
    "position_sizing": False,    # 禁用动态仓位管理
    "max_trade_amount": 0.05,    # 单次交易最大金额比例(相对于初始资金)
    "early_stop_loss_threshold": 0.15,  # 提前停止的损失阈值（15%）
    "early_stop_enabled": True,  # 是否启用提前停止机制
    
    # 状态空间配置
    "use_technical_indicators": True,  # 是否使用技术指标
    "normalize_state": True,     # 是否归一化状态
    "include_position": True,    # 是否包含当前持仓信息
    
    # 奖励配置
    "reward_type": "compound",   # 奖励函数类型: compound（组合奖励）
    "reward_scaling": 0.01,      # 奖励缩放系数
    "penalize_inaction": True,   # 是否惩罚不交易行为
    "inaction_penalty": -0.1,    # 不交易惩罚系数
    
    # 交易控制
    "min_trade_interval": 5,     # 最小交易间隔（步数）
    "max_trades_per_episode": 20,  # 每个回合最大交易次数
    
    # 奖励权重配置
    "reward_weights": {
        "profit": 1.0,              # 单次交易收益
        "cumulative_return": 2.0,   # 累积收益
        "risk_adjusted": 1.5,       # 风险调整后收益
        "drawdown": 1.0,            # 回撤惩罚
        "trade_frequency": 0.5,     # 交易频率控制
        "inaction": 0.3,            # 不交易惩罚
        "trend_follow": 0.8,        # 趋势跟随奖励
        "consecutive_buy": 1.5      # 连续买入惩罚
    },
    
    # 详细奖励配置参数
    "reward_config": {
        "max_reward_limit": 0.2,           # 限制最大单步收益率为20%
        "max_drawdown_penalty": -0.5,      # 最大回撤惩罚
        "inaction_base_penalty": -0.05,    # 不交易基础惩罚
        "inaction_time_penalty": -0.02,    # 不交易时间惩罚
        "trend_misalign_penalty": -0.1,    # 趋势不一致惩罚
        "trend_follow_reward": 0.15,       # 趋势跟随奖励
        "frequent_trade_penalty": -0.1,    # 频繁交易惩罚
        "position_holding_penalty": -0.1,  # 长时间持仓惩罚
        "consecutive_buy_base_penalty": -0.2,  # 连续买入基础惩罚
        "trade_interval_threshold": 10     # 交易间隔阈值，超过此值增加惩罚
    }
}

# DQN模型配置
DQN_CONFIG = {
    # 网络结构
    "hidden_layers": [256, 192, 128, 64],  # 4层网络结构
    "activation": "relu",        # 激活函数
    "learning_rate": 0.0001,     # 学习率
    
    # 训练参数
    "batch_size": 96,            # 批次大小
    "buffer_size": 5000,         # 经验回放缓冲区大小(从20000减少到5000)
    "gamma": 0.98,               # 折扣因子
    "update_target_every": 100,  # 目标网络更新频率
    "epsilon_start": 1.0,        # 初始探索率
    "epsilon_end": 0.01,         # 最终探索率
    "epsilon_decay": 0.995,      # 探索率衰减因子
    "double_dqn": True,          # 使用Double DQN算法
    
    # 训练控制
    "max_episodes": 1000,        # 最大训练回合数
    "eval_frequency": 20,        # 评估频率
    "save_frequency": 50,        # 保存频率
    
    # 风险控制
    "max_drawdown_limit": 0.15,  # 最大回撤限制（15%）
    "position_scaling": True,    # 是否启用动态仓位调整
    "risk_aversion": 0.5,       # 风险厌恶系数
}

# PPO模型配置
PPO_CONFIG = {
    # 网络结构
    "hidden_layers": [128, 64],  # 隐藏层神经元数
    "activation": "tanh",        # 激活函数
    "learning_rate": 0.0003,     # 学习率
    
    # 训练参数
    "gamma": 0.99,               # 折扣因子
    "lambda_gae": 0.95,          # GAE参数
    "clip_ratio": 0.2,           # PPO剪裁比例
    "entropy_coef": 0.01,        # 熵系数
    "value_coef": 0.5,           # 价值系数
    "epochs_per_update": 4,      # 每次更新的训练轮数
    "batch_size": 64,            # 批次大小
    
    # 训练控制
    "max_episodes": 1000,        # 最大训练回合数
    "eval_frequency": 20,        # 评估频率
    "save_frequency": 50,        # 保存频率
}

# A2C模型配置
A2C_CONFIG = {
    # 网络结构
    "hidden_layers": [128, 64],  # 隐藏层神经元数
    "activation": "relu",        # 激活函数
    "learning_rate": 0.0005,     # 学习率
    
    # 训练参数
    "gamma": 0.99,               # 折扣因子
    "entropy_coef": 0.01,        # 熵系数
    "value_coef": 0.5,           # 价值系数
    "max_grad_norm": 0.5,        # 梯度剪裁值
    
    # 训练控制
    "max_episodes": 1000,        # 最大训练回合数
    "eval_frequency": 20,        # 评估频率
    "save_frequency": 50,        # 保存频率
}

# 模型保存路径
MODEL_SAVE_PATH = "saved_models/" 