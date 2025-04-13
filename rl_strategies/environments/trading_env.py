"""
交易环境基类

定义了强化学习交易环境的基本结构和功能
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
import random


class TradingEnv(gym.Env):
    """
    交易环境基类
    实现了基本的交易环境逻辑，包括状态空间、动作空间和奖励计算
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size=None,
        initial_balance=None,
        transaction_fee=None,
        reward_type=None,
        use_technical_indicators=None,
        include_position=None,
        penalize_inaction=None,
        max_position_size=None,
        base_position_size=None,
        position_sizing=None,
        fixed_trade_amount=None,
        max_trade_amount=None,
        max_episode_steps=None,
        early_stop_loss_threshold=None,
        early_stop_enabled=None,
        min_trade_interval=None,
        max_trades_per_episode=None,
        verbose=None,
        env_type='training'
    ):
        """
        初始化交易环境
        
        参数:
            df: 价格数据
            window_size: 观察窗口大小
            initial_balance: 初始资金
            transaction_fee: 交易费率
            reward_type: 奖励类型 ('profit', 'sharpe', 'sortino', 'compound')
            use_technical_indicators: 是否使用技术指标
            include_position: 是否包含持仓信息
            penalize_inaction: 是否惩罚不行动
            max_position_size: 最大仓位比例
            base_position_size: 基础仓位比例
            position_sizing: 是否启用仓位管理
            fixed_trade_amount: 固定每次交易金额
            max_trade_amount: 单次交易最大金额比例(相对于初始资金)
            max_episode_steps: 最大步数
            early_stop_loss_threshold: 提前停止的损失阈值（占初始资金的比例）
            early_stop_enabled: 是否启用提前停止机制
            min_trade_interval: 最小交易间隔（步数）
            max_trades_per_episode: 每个回合最大交易次数
            verbose: 是否输出详细日志
            env_type: 环境类型 ('training' 或 'evaluation')
        """
        super(TradingEnv, self).__init__()
        
        # 导入默认环境配置
        from rl_strategies.config import DEFAULT_ENV_CONFIG
        
        # 数据预处理
        self.df = df.copy()
        
        # 设置环境参数，优先使用传入参数，否则使用默认配置
        self.window_size = window_size if window_size is not None else DEFAULT_ENV_CONFIG['window_size']
        self.initial_balance = initial_balance if initial_balance is not None else DEFAULT_ENV_CONFIG['initial_balance']
        self.transaction_fee = transaction_fee if transaction_fee is not None else DEFAULT_ENV_CONFIG['transaction_fee']
        self.reward_type = reward_type if reward_type is not None else DEFAULT_ENV_CONFIG['reward_type']
        self.use_technical_indicators = use_technical_indicators if use_technical_indicators is not None else DEFAULT_ENV_CONFIG['use_technical_indicators']
        self.include_position = include_position if include_position is not None else DEFAULT_ENV_CONFIG['include_position']
        self.penalize_inaction = penalize_inaction if penalize_inaction is not None else DEFAULT_ENV_CONFIG['penalize_inaction']
        self.max_position_size = max_position_size if max_position_size is not None else DEFAULT_ENV_CONFIG['max_position_size']
        self.base_position_size = base_position_size if base_position_size is not None else DEFAULT_ENV_CONFIG['base_position_size']
        self.position_sizing = position_sizing if position_sizing is not None else DEFAULT_ENV_CONFIG['position_sizing']
        self.fixed_trade_amount = fixed_trade_amount if fixed_trade_amount is not None else DEFAULT_ENV_CONFIG['fixed_trade_amount']
        self.max_trade_amount = max_trade_amount if max_trade_amount is not None else DEFAULT_ENV_CONFIG['max_trade_amount']
        self.early_stop_loss_threshold = early_stop_loss_threshold if early_stop_loss_threshold is not None else DEFAULT_ENV_CONFIG['early_stop_loss_threshold']
        self.early_stop_enabled = early_stop_enabled if early_stop_enabled is not None else DEFAULT_ENV_CONFIG['early_stop_enabled']
        self.min_trade_interval = min_trade_interval if min_trade_interval is not None else DEFAULT_ENV_CONFIG['min_trade_interval']
        self.max_trades_per_episode = max_trades_per_episode if max_trades_per_episode is not None else DEFAULT_ENV_CONFIG['max_trades_per_episode']
        self.verbose = verbose if verbose is not None else True  # 默认为True，配置文件中可能没有这个参数
        
        # 新增缺失的变量
        self.min_trade_amount = 0.01  # 最小交易数量，用于计算资金是否足够买入
        self.max_position_limit = self.max_position_size  # 最大持仓限制，避免过度买入
        
        # 如果使用技术指标，添加到数据中
        if self.use_technical_indicators:
            self.df = self._add_indicators(self.df)
        
        # 提前停止标志
        self.stopped_early = False
        
        # 交易记录增强
        self.last_buy_price = 0.0           # 记录最后一次买入价格
        self.last_buy_cost = 0.0            # 记录最后一次买入的总成本（含手续费）
        self.last_buy_shares = 0.0          # 记录最后一次买入的股数
        self.last_buy_timestamp = None      # 记录最后一次买入的时间戳
        self.avg_entry_price = 0.0          # 持仓均价（包含手续费）
        self.total_cost = 0.0               # 总买入成本（包含手续费）
        
        # 复合奖励权重 (当reward_type='compound'时使用)
        # 使用默认值，这些会在UI中设置，通过trainer传递过来
        self.reward_weights = DEFAULT_ENV_CONFIG.get('reward_weights', {
            'profit': 1.0,              # 单次交易收益
            'cumulative_return': 2.0,   # 累积收益
            'risk_adjusted': 1.5,       # 风险调整后收益
            'drawdown': 1.0,            # 回撤惩罚
            'trade_frequency': 0.5,     # 交易频率控制
            'inaction': 0.3,            # 不交易惩罚
            'trend_follow': 0.8,        # 趋势跟随奖励
            'consecutive_buy': 1.5      # 连续买入惩罚
        })
        
        # 设置最大步数，如果未提供则使用数据长度
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else (len(self.df) - self.window_size - 1)
        
        # 特征列
        self.features = ['open', 'high', 'low', 'close', 'volume']
        if use_technical_indicators:
            self.features += ['sma7', 'sma25', 'rsi', 'bb_middle', 'bb_upper', 'bb_lower', 'macd', 'macd_signal']
            
        # 动作空间: 0 (卖出), 1 (持有), 2 (买入)
        self.action_space = gym.spaces.Discrete(3)
        
        # 状态空间: 特征 × 窗口大小 + 额外状态
        feature_dim = len(self.features) * self.window_size
        extra_dim = 2 if include_position else 0  # 持仓价值和余额
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim + extra_dim,), dtype=np.float32)
        
        # 初始化状态变量
        self.current_step = None
        self.balance = None
        self.position = None
        self.position_value = None
        self.total_profit = None
        self.trade_count = None
        self.buy_hold_value = None  # 买入持有策略的价值
        self.trade_history = []     # 交易历史记录
        self.drawdown_history = []  # 回撤历史
        self.max_drawdown = 0       # 最大回撤
        self.unrealized_profit = 0  # 未实现收益
        self.last_action_step = 0   # 上次交易步数
        self.holding_steps = 0      # 持仓步数
        self.trend_direction = 0    # 当前市场趋势方向
        self.episode_step_count = 0  # 当前回合的步数
        
        # 初始化账户跟踪相关属性
        self.portfolio_value_tracker = []  # 资产价值追踪
        self.balance_tracker = []  # 余额追踪
        self.position_value_tracker = []  # 持仓价值追踪
        self.step_tracker = []  # 步骤追踪
        self.account_history = []  # 账户历史记录
        
        # 如果使用复合奖励，还需要额外的奖励配置参数
        self.reward_config = DEFAULT_ENV_CONFIG.get('reward_config', {
            'max_reward_limit': 0.2,          # 限制最大单步收益率为20%
            'max_drawdown_penalty': -0.5,     # 最大回撤惩罚
            'inaction_base_penalty': -0.05,   # 不交易基础惩罚
            'inaction_time_penalty': -0.02,   # 不交易时间惩罚
            'trend_misalign_penalty': -0.1,   # 趋势不一致惩罚
            'trend_follow_reward': 0.15,      # 趋势跟随奖励
            'frequent_trade_penalty': -0.1,   # 频繁交易惩罚
            'position_holding_penalty': -0.1, # 长时间持仓惩罚
            'consecutive_buy_base_penalty': -0.2, # 连续买入基础惩罚
            'trade_interval_threshold': 10    # 交易间隔阈值，超过此值增加惩罚
        })
        
        # 重置环境
        self.reset()
        
        # 记录环境类型
        self.env_type = env_type
        
        # 根据环境类型创建不同的交易记录容器
        if env_type == 'training':
            self.train_transaction_history = []  # 训练环境的交易记录
        else:
            self.eval_transaction_history = []   # 评估环境的交易记录
        
        # 回撤跟踪
        self.drawdown_start = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # 持仓跟踪
        self.max_position = self.initial_balance * 0.5  # 默认最大持仓为50%资金
        self.entry_price = 0.0  # 入场价格
        self.holding_steps = 0  # 持仓时间
        self.position_profits = []  # 每次交易的收益
        
        # 价值追踪
        self.portfolio_values = []  # 投资组合价值历史
        self.max_portfolio_value = self.initial_balance  # 最大投资组合价值
        self.min_portfolio_value = self.initial_balance  # 最小投资组合价值
        self.initial_price = self.df.iloc[0]['close']  # 初始价格
        self.buy_hold_value = self.initial_balance  # 买入持有策略价值
        self.trade_history = []  # 交易历史（兼容旧接口）
        self.drawdown_history = []  # 回撤历史
        
        # 手续费记录
        self.fees_paid = 0.0  # 已支付的手续费
        
        # 市场方向
        self.trend_direction = 0  # 0: 横盘, 1: 上升, -1: 下降
        
        # 重置环境
        self.current_step = 0
    
    def reset(self, **kwargs):
        """重置环境状态"""
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        self.total_pnl = 0
        self.consecutive_buy_count = 0  # 确保重置连续买入计数
        self.last_action = 1  # 1表示持有，即默认的初始动作
        self.last_buy_price = 0  # 上次买入价格
        self.max_position = 0  # 记录最大持仓量
        self.max_portfolio_value = self.initial_balance
        self.min_portfolio_value = self.initial_balance
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.drawdown_start = self.initial_balance
        self.portfolio_values = [self.initial_balance]  # 记录资产价值历史
        self.transaction_history = []  # 交易历史
        self.trade_count = 0  # 交易次数
        self.fees_paid = 0  # 支付的手续费总额
        self.last_trade_step = 0  # 上次交易的步骤
        self.inaction_steps = 0  # 未采取交易动作的连续步数
        self.current_step = self.window_size - 1  # 初始步骤
        self.done = False
        self.truncated = False
        self.entry_price = 0  # 入场价格
        self.cumulative_reward = 0  # 重置累计奖励值
        
        # 设置初始价格，用于买入持有策略计算
        self.initial_price = self.df.iloc[self.current_step]['close']
        
        # 初始化账户历史追踪器
        self.portfolio_value_tracker = [self.initial_balance]  # 资产价值追踪
        self.balance_tracker = [self.initial_balance]  # 余额追踪
        self.position_value_tracker = [0]  # 持仓价值追踪
        self.step_tracker = [self.current_step]  # 步骤追踪
        self.account_history = []  # 账户历史记录
        self.episode_step_count = 0  # 重置步数计数

        # 打印初始状态
        print(f"环境已重置 - 初始资金: {self.initial_balance}, 持仓: {self.position}, 连续买入计数: {self.consecutive_buy_count}")
        
        # 构建初始状态
        state = self._get_observation()
        info = {
            'initial_balance': self.initial_balance,
            'portfolio_value': self.initial_balance,
            'position': self.position,
            'step': 0
        }
        
        # 返回观察和信息，符合gymnasium标准
        return (state, info)
    
    def _get_observation(self):
        """获取当前观察状态"""
        # 获取当前窗口的特征数据
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        
        # 提取特征列数据
        window_data = self.df.iloc[start:end][self.features].values
        
        # 归一化
        window_data = self._normalize_window(window_data)
        
        # 展平窗口数据
        observation = window_data.flatten()
        
        # 如果包含持仓信息，添加到观察中
        if self.include_position:
            # 归一化持仓价值和余额
            position_value_normalized = self.position_value / self.initial_balance
            balance_normalized = self.balance / self.initial_balance
            
            # 添加到观察中
            observation = np.append(observation, [position_value_normalized, balance_normalized])
        
        return observation
    
    def _normalize_window(self, window):
        """归一化窗口数据，使用MinMax缩放"""
        # 对于价格和成交量等列，分别进行归一化
        normalized_window = np.zeros_like(window)
        
        for i in range(window.shape[1]):
            # 获取当前特征列
            column = window[:, i]
            
            # 计算最大最小值，避免除以零
            column_min = np.min(column)
            column_max = np.max(column)
            
            if column_max > column_min:
                # 归一化到[0, 1]范围
                normalized_window[:, i] = (column - column_min) / (column_max - column_min)
            else:
                # 如果最大值等于最小值，设为0.5
                normalized_window[:, i] = 0.5
        
        return normalized_window
    
    def _update_trend_direction(self):
        """更新当前市场趋势方向"""
        if self.current_step >= 10:
            # 使用简单的短期（10个周期）移动平均线判断趋势
            current_ma = self.df.iloc[self.current_step-9:self.current_step+1]['close'].mean()
            previous_ma = self.df.iloc[self.current_step-10:self.current_step]['close'].mean()
            
            if current_ma > previous_ma:
                self.trend_direction = 1  # 上升趋势
            elif current_ma < previous_ma:
                self.trend_direction = -1  # 下降趋势
            else:
                self.trend_direction = 0  # 横盘
    
    def step(self, action):
        """
        执行一步交易
        
        参数:
            action: 交易动作 (0: 卖出, 1: 持有, 2: 买入)
        
        返回:
            observation: 观察状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 增加当前回合的相对步数
        self.episode_step_count += 1
        step_count = self.episode_step_count  # 从1开始计数，更直观
        
        # 初始化变量
        done = False
        truncated = False
        early_stop = False  # 修复：初始化early_stop变量
        
        # 检查是否有持仓，如果没有持仓且动作为卖出，则改为持有
        original_action = action
        if action == 0 and self.position <= 0:
            action = 1  # 改为持有
        
        # 记录总资产变化
        previous_portfolio_value = self.balance + self.position_value
        
        # 执行交易动作
        self._take_action(action)
        
        # 更新当前步数
        self.current_step += 1
        
        # 检查是否已经超出数据范围
        if self.current_step >= len(self.df):
            print(f"警告: 当前步数 {self.current_step} 已超出数据范围 {len(self.df)}，强制结束回合")
            return self._get_observation(), 0, True, False, {'portfolio_value': previous_portfolio_value, 'stopped_early': True}
        
        # 更新买入持有策略的价值（作为比较基准）
        current_price = self.df.iloc[self.current_step]['close']
        initial_shares = self.initial_balance / self.initial_price
        self.buy_hold_value = initial_shares * current_price
        
        # 更新账户状态历史
        self._track_account_status()
        
        # 计算当前总资产
        current_portfolio_value = self.balance + self.position_value
        
        # 计算当前总收益率
        total_return_pct = (current_portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        # 计算当前最大回撤
        if current_portfolio_value > 0:
            drawdown = 1 - current_portfolio_value / self.initial_balance if current_portfolio_value < self.initial_balance else 0  # 修正：只有当前资产小于初始资产时才有回撤
            self.drawdown_history.append(drawdown)
            self.max_drawdown = max(self.max_drawdown, drawdown)  # 更新最大回撤
        
        # 更新持仓步数
        if self.position > 0:
            self.holding_steps += 1
        else:
            self.holding_steps = 0
            
        # 更新趋势方向
        self._update_trend_direction()
        
        # 准备观察状态
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._calculate_reward(previous_portfolio_value, current_portfolio_value, action)
        
        # *** 强制环境不在前400步结束 ***
        # 这是为了确保每个回合至少执行400步训练
        force_continue = step_count < 400
        
        # 提前停止条件检查 - 修改为使用truncated而不是done来处理提前停止
        # 只有在不强制继续的情况下才检查提前停止条件
        if self.early_stop_enabled and not force_continue:
            # 提前停止条件：如果总资产低于初始资产的门限值
            # 更新计算方式，确保正确计算损失率
            current_loss_pct = (self.initial_balance - current_portfolio_value) / self.initial_balance
            
            if current_loss_pct > self.early_stop_loss_threshold:
                # 添加随机因素，避免总是在特定步数停止
                if np.random.random() > 0.2:  # 80%的概率实际截断
                    truncated = True  # 使用truncated而不是立即设置done
                    early_stop = True
                    self.stopped_early = True
                    
                    # 添加详细日志
                    print(f"\n===== 提前停止 =====")
                    print(f"时间: {self.df.iloc[self.current_step].name}")
                    print(f"当前总资产: {current_portfolio_value:.2f} (初始: {self.initial_balance:.2f})")
                    print(f"当前损失率: {current_loss_pct*100:.2f}% (阈值: {self.early_stop_loss_threshold*100:.2f}%)")
                    print(f"回合内交易次数: {self.trade_count}")
                    print(f"当前回合步数: {step_count}")
                    print(f"========================")
                else:
                    print(f"损失率达到阈值但随机决定继续: 损失率={current_loss_pct*100:.2f}%, 步数={step_count}")
        
        # 正常回合结束条件：达到最大步数或数据末尾
        # 只有在不强制继续的情况下才检查正常结束条件
        max_allowed_step = min(self.window_size + self.max_episode_steps, len(self.df) - 1)
        
        # 修正步数计算逻辑 - 使用相对步数而不是绝对位置
        if (step_count >= self.max_episode_steps) and not force_continue:
            done = True
        
        # 组装额外信息
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'position_value': self.position_value,
            'position': self.position,
            'total_return_pct': total_return_pct,
            'drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'price': current_price,
            'stopped_early': early_stop,
            'step': step_count,
            'total_steps': self.max_episode_steps,
            'current_step': self.current_step,
            'max_step': max_allowed_step,
            'force_continue': force_continue,
            'loss_pct': (self.initial_balance - current_portfolio_value) / self.initial_balance * 100  # 明确添加损失百分比
        }
        
        # ================== 开始调试日志 ==================
        if step_count % 20 == 0 or done or truncated or step_count == 101:
            print(f"DEBUG-ENV-STEP[{step_count}]: 步骤完成，done={done}, truncated={truncated}, 强制继续={force_continue}")
        # ================== 结束调试日志 ==================
        
        return observation, reward, done, truncated, info
    
    def _take_action(self, action):
        """
        执行交易动作
        
        参数:
            action: 交易动作 (0: 卖出, 1: 持有, 2: 买入)
            
        返回:
            action_result: 动作结果
        """
        action_result = 'success'
        current_price = self.df.iloc[self.current_step]['close']
        transaction_executed = False
        transaction_type = None
        transaction_amount = 0
        transaction_reason = ""
        
        print(f"\n====== 执行交易动作 {action} ======")
        print(f"当前步数: {self.current_step}")
        print(f"当前价格: {current_price}")
        print(f"当前持仓: {self.position}")
        print(f"当前余额: {self.balance}")
        
        # 卖出操作
        if action == 0:
            print("\n----- 尝试执行卖出操作 -----")
            if self.position > 0:  # 有持仓才能卖出
                old_position = self.position
                
                # 计算卖出价值（扣除手续费），使用8位小数，避免精度损失
                sell_amount = round(self.position * current_price, 8)  # 先计算卖出金额
                sell_fee = round(sell_amount * self.transaction_fee, 8)  # 计算手续费
                if sell_fee == 0 and self.transaction_fee > 0:  # 如果手续费被四舍五入为0，但费率不为0
                    sell_fee = round(sell_amount * self.transaction_fee, 10)  # 使用更高精度
                sell_value = round(sell_amount - sell_fee, 8)  # 实际卖出价值为卖出金额减去手续费
                
                print(f"卖出计算详情:")
                print(f"- 卖出数量: {old_position}")
                print(f"- 卖出金额(未扣费): {sell_amount}")
                print(f"- 手续费: {sell_fee}")
                print(f"- 实际收入: {sell_value}")
                
                # 更新手续费总额
                self.fees_paid = round(self.fees_paid + sell_fee, 8)
                
                # 计算利润（使用持仓均价）
                if self.avg_entry_price > 0:
                    # 计算买入成本（基于持仓均价，包含手续费）
                    buy_cost = round(self.position * self.avg_entry_price, 8)
                    # 计算实际利润
                    total_profit = round(sell_value - buy_cost, 8)
                    profit_pct = round(total_profit / buy_cost * 100, 8)
                    # 更新总利润
                    self.total_profit = round(self.total_profit + total_profit, 8) if self.total_profit else total_profit
                    print(f"利润计算详情:")
                    print(f"- 买入成本: {buy_cost}")
                    print(f"- 实际利润: {total_profit}")
                    print(f"- 利润率: {profit_pct}%")
                    print(f"- 累计利润: {self.total_profit}")
                    
                    # 添加正向利润奖励的打印
                    if total_profit > 0:
                        profit_base_reward = self.reward_config.get('profit_base_reward', 0.05)
                        print(f"- 正向利润奖励: +{profit_base_reward:.4f} (将在奖励计算时添加)")
                    
                    # 保存卖出交易的关键信息，用于奖励计算
                    self.last_sell_position = self.position  # 记录卖出前的持仓量
                    self.last_sell_price = current_price  # 记录卖出价格
                    self.last_sell_avg_entry = self.avg_entry_price  # 记录卖出前的平均入场价
                    
                    # 计算交易收益率（考虑手续费）用于奖励计算
                    sell_price_after_fee = current_price * (1 - self.transaction_fee)  # 考虑卖出手续费
                    entry_price_with_fee = self.avg_entry_price * (1 + self.transaction_fee)  # 考虑买入手续费
                    trade_return = (sell_price_after_fee - entry_price_with_fee) / entry_price_with_fee
                    
                    # 保存交易收益率和实际利润，供奖励计算使用
                    self.last_trade_return = trade_return
                    self.last_actual_profit = total_profit
                
                # 更新余额和持仓
                old_balance = self.balance
                self.balance = round(self.balance + sell_value, 8)
                self.position = 0
                self.position_value = 0
                self.total_cost = 0
                self.avg_entry_price = 0
                
                print(f"账户更新详情:")
                print(f"- 原余额: {old_balance}")
                print(f"- 新余额: {self.balance}")
                print(f"- 持仓已清空")
                
                # 更新交易记录
                transaction_executed = True
                transaction_type = 'sell'
                transaction_amount = old_position
                
                # 重置连续买入计数
                self.consecutive_buy_count = 0
                
                print(f"\n交易执行成功 - 卖出 {old_position} 单位，价格: {current_price:.2f}，收入: {sell_value:.2f}（手续费: {sell_fee:.2f}）")
                self.last_action = 0
                self.last_trade_step = self.current_step
                self.inaction_steps = 0
                self.trade_count += 1
            else:
                print(f"卖出失败 - 当前无持仓，无法卖出")
                action_result = 'no_position'
        
        # 买入操作
        elif action == 2:
            if self.balance > 0:  # 有余额才能买入
                # 计算可买入的数量（考虑手续费）
                affordable_amount = self.balance / (current_price * (1 + self.transaction_fee))
                
                # 限制单次买入的数量不超过最大交易量或余额
                max_amount = min(self.max_trade_amount, affordable_amount)
                
                if max_amount < self.min_trade_amount:
                    # 资金不足以购买最小交易量
                    print(f"动作: 买入失败 - 资金不足! 当前余额: {self.balance:.2f}，最小购买所需: {self.min_trade_amount * current_price * (1 + self.transaction_fee):.2f}")
                    action_result = 'insufficient_funds'
                else:
                    # 获取买入数量
                    amount_to_buy = self._calculate_buy_amount(max_amount)
                    
                    # 计算买入成本（包括手续费），使用8位小数，避免精度损失
                    buy_amount = round(amount_to_buy * current_price, 8)  # 先计算买入金额
                    buy_fee = round(buy_amount * self.transaction_fee, 8)  # 计算手续费，确保不会为0
                    if buy_fee == 0 and self.transaction_fee > 0:  # 如果手续费被四舍五入为0，但费率不为0
                        buy_fee = round(buy_amount * self.transaction_fee, 10)  # 使用更高精度
                    cost = round(buy_amount + buy_fee, 8)  # 总成本为买入金额加手续费
                    
                    # 更新手续费总额
                    self.fees_paid = round(self.fees_paid + buy_fee, 8)
                    
                    # 更新余额和持仓
                    old_balance = self.balance
                    self.balance = round(self.balance - cost, 8)
                    
                    # 更新持仓均价
                    old_total_cost = round(self.total_cost, 8)
                    old_position = round(self.position, 8)
                    
                    # 计算新的总成本和总持仓
                    self.total_cost = round(old_total_cost + cost, 8)
                    self.position = round(old_position + amount_to_buy, 8)
                    
                    # 计算新的持仓均价（包含手续费）
                    if self.position > 0:
                        self.avg_entry_price = round(self.total_cost / self.position, 8)
                    
                    self.position_value = round(self.position * current_price, 8)
                    
                    # 更新最大持仓记录
                    self.max_position = max(self.max_position, self.position)
                    
                    # 如果是首次买入，记录入场价格
                    if self.entry_price == 0:
                        self.entry_price = current_price
                    
                    # 如果连续买入，更新最后买入价格
                    self.last_buy_price = current_price
                    
                    # 更新交易记录
                    transaction_executed = True
                    transaction_type = 'buy'
                    transaction_amount = amount_to_buy
                    
                    # 更新连续买入计数
                    self.consecutive_buy_count += 1
                    
                    print(f"动作: 买入 - 购买 {amount_to_buy:.4f} 单位，价格: {current_price:.2f}，成本: {cost:.2f}（手续费: {buy_fee:.2f}）")
                    print(f"资金状态: 余额: {self.balance:.2f}，持仓: {self.position:.4f}，持仓价值: {self.position_value:.2f}，连续买入计数: {self.consecutive_buy_count}")
                    
                    self.last_action = 2
                    self.last_trade_step = self.current_step
                    self.inaction_steps = 0
                    self.trade_count += 1
            else:
                # 余额不足
                print(f"动作: 买入失败 - 资金不足! 当前余额: {self.balance:.2f}")
                action_result = 'insufficient_funds'
        
        else:  # 持有
            self.inaction_steps += 1
            print(f"动作: 持有 - 当前持仓: {self.position:.4f}，价值: {self.position_value:.2f}，余额: {self.balance:.2f}")
            self.last_action = 1
        
        # 计算当前资产组合价值
        portfolio_value = self.balance + self.position_value
        
        # 更新资产价值历史
        self.portfolio_values.append(portfolio_value)
        
        # 更新最大和最小资产价值记录
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        self.min_portfolio_value = min(self.min_portfolio_value, portfolio_value)
        
        # 计算回撤
        if portfolio_value < self.drawdown_start:
            current_drawdown = (self.drawdown_start - portfolio_value) / self.drawdown_start
            if current_drawdown > self.current_drawdown:
                self.current_drawdown = current_drawdown
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
        else:
            self.drawdown_start = portfolio_value
            self.current_drawdown = 0
        
        # 记录交易
        if transaction_executed:
            timestamp = str(self.df.index[self.current_step])
            transaction = {
                'step': self.current_step,
                'price': current_price,
                'type': transaction_type,
                'action': '买入' if transaction_type == 'buy' else '卖出',
                'amount': transaction_amount,
                'timestamp': timestamp,
                'balance': self.balance,
                'position': self.position,
                'portfolio_value': portfolio_value,
                'reason': transaction_reason,
                'last_buy_price': self.last_buy_price,  # 添加最后一次买入价格字段
                'total_value': portfolio_value  # 确保总价值存在
            }
            
            # 添加交易金额和手续费信息
            if transaction_type == 'buy':
                cost = transaction_amount * current_price
                fee = cost * self.transaction_fee
                transaction['buy_amount'] = cost
                transaction['buy_fee'] = fee
                
                # 确保交易记录完整，打印调试信息
                print(f"DEBUG-TRANSACTION: 买入 {transaction_amount:.6f} ETH，总金额: {cost:.2f}，手续费: {fee:.2f}")
            elif transaction_type == 'sell':
                sell_value = transaction_amount * current_price * (1 - self.transaction_fee)
                fee = transaction_amount * current_price * self.transaction_fee
                transaction['sell_value'] = sell_value
                transaction['sell_fee'] = fee
                
                # 计算利润（使用持仓均价）
                if self.avg_entry_price > 0:
                    # 计算买入成本（基于持仓均价）
                    buy_cost = transaction_amount * self.avg_entry_price
                    # 计算卖出收入（扣除手续费）
                    sell_income = transaction_amount * current_price * (1 - self.transaction_fee)
                    # 计算实际利润
                    profit = round(sell_income - buy_cost, 8)
                    # 计算利润百分比（基于持仓均价）
                    profit_pct = round((sell_income - buy_cost) / buy_cost * 100, 4)
                    transaction['profit'] = profit
                    transaction['profit_pct'] = profit_pct
                    transaction['avg_entry_price'] = self.avg_entry_price  # 添加持仓均价到交易记录
                    
                    # 确保交易记录完整，打印调试信息
                    print(f"DEBUG-TRANSACTION: 卖出 {transaction_amount:.6f} ETH，总金额: {sell_value:.2f}，手续费: {fee:.2f}，利润: {profit:.2f}({profit_pct:.2f}%)")
            
            # 打印完整交易记录
            print(f"DEBUG-TRANSACTION-RECORD: {transaction}")
            
            # 保存到通用交易历史
            self.transaction_history.append(transaction)
            
            # 根据环境类型保存到相应的交易历史
            if self.env_type == 'training':
                if hasattr(self, 'train_transaction_history'):
                    self.train_transaction_history.append(transaction)
                    print(f"DEBUG-ENV: 添加交易记录到训练环境历史，当前共 {len(self.train_transaction_history)} 条记录")
            else:
                if hasattr(self, 'eval_transaction_history'):
                    self.eval_transaction_history.append(transaction)
                    print(f"DEBUG-ENV: 添加交易记录到评估环境历史，当前共 {len(self.eval_transaction_history)} 条记录")
            
            # 为保持向后兼容
            if hasattr(self, 'trade_history'):
                self.trade_history.append(transaction)
        
        return action_result
    
    def _calculate_drawdown(self, current_value):
        """计算当前回撤"""
        # 更新历史最高价值
        if not self.drawdown_history:
            self.drawdown_history.append(current_value)
        
        peak = max(self.drawdown_history)
        self.drawdown_history.append(current_value)
        
        # 计算当前回撤
        if current_value < peak:
            drawdown = (peak - current_value) / peak
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def _calculate_reward(self, previous_portfolio_value, current_portfolio_value, action):
        """计算奖励函数，基于单次交易金额的收益率"""
        # 从配置中获取奖励放大因子
        reward_amplifier = self.reward_config.get('reward_amplifier', 20.0)
        
        # 从配置中获取成功交易基础奖励
        profit_base_reward = self.reward_config.get('profit_base_reward', 0.05)
        
        # 获取当前价格
        current_price = self.df.iloc[self.current_step]['close']
        
        # 初始化总奖励值
        total_reward = 0.0
        
        # 检查是否为卖出交易
        if action == 0 and hasattr(self, 'last_actual_profit'):
            # 收益率奖励
            if hasattr(self, 'last_trade_return'):
                trade_return = self.last_trade_return
                # 基于收益率的奖励
                return_reward = trade_return * reward_amplifier
                
                # 确保小的正收益率产生最小正奖励
                if trade_return > 0 and return_reward < 0.001:
                    return_reward = 0.001
                
                # 保留3位小数精度
                return_reward = round(return_reward, 3)
                
                total_reward += return_reward
                print(f"\n=== 卖出交易奖励计算 ===")
                print(f"收益率: {trade_return*100:.4f}%")
                if return_reward >= 0:
                    print(f"收益率奖励值: +{return_reward:.3f} (收益率 × 放大因子 {reward_amplifier})")
                else:
                    print(f"收益率惩罚值: {return_reward:.3f} (收益率 × 放大因子 {reward_amplifier})")
                
                # 基础正向利润奖励
                if self.last_actual_profit > 0:
                    profit_base_reward = round(profit_base_reward, 3)  # 保留3位小数精度
                    total_reward += profit_base_reward
                    print(f"基础成功奖励值: +{profit_base_reward:.3f} (盈利交易)")
                
                # 趋势方向奖励
                if self.trend_direction * trade_return > 0:
                    trend_reward = self.reward_config.get('trend_follow_reward', 0.1)
                    trend_reward = round(trend_reward, 3)  # 保留3位小数精度
                    total_reward += trend_reward
                    print(f"趋势跟随奖励值: +{trend_reward:.3f}")
                
                # 打印总奖励
                if total_reward >= 0:
                    print(f"总奖励值: +{total_reward:.3f}\n")
                else:
                    print(f"总惩罚值: {total_reward:.3f}\n")
        
        # 买入奖励
        elif action == 2:
            # 买入基础奖励
            base_reward = 0.01
            base_reward = round(base_reward, 3)  # 保留3位小数精度
            total_reward += base_reward
            
            # 输出买入奖励信息
            print(f"\n=== 买入交易奖励计算 ===")
            print(f"买入基础奖励值: +{base_reward:.3f}")
            
            # 根据趋势方向给予奖励
            if self.trend_direction == 1:
                trend_reward = self.reward_config.get('trend_follow_reward', 0.3) / 10
                trend_reward = round(trend_reward, 3)  # 保留3位小数精度
                total_reward += trend_reward
                print(f"上升趋势买入奖励值: +{trend_reward:.3f}")
            
            # 连续买入惩罚
            if self.consecutive_buy_count > 1:
                consecutive_penalty = self.reward_config.get('consecutive_buy_base_penalty', -0.05) * (self.consecutive_buy_count - 1)
                consecutive_penalty = round(consecutive_penalty, 3)  # 保留3位小数精度
                total_reward += consecutive_penalty
                print(f"连续买入惩罚值: {consecutive_penalty:.3f} (连续{self.consecutive_buy_count}次)")
            
            # 打印小计
            if total_reward >= 0:
                print(f"总奖励值: +{total_reward:.3f}\n")
            else:
                print(f"总惩罚值: {total_reward:.3f}\n")
        
        # 持有奖励
        else:  # action == 1
            # 如果持有仓位，根据未实现收益率给予奖励
            if self.position > 0 and self.last_buy_price > 0:
                unrealized_return = (current_price - self.last_buy_price) / self.last_buy_price
                position_ratio = self.position_value / self.initial_balance
                
                # 放大未实现收益的奖励
                total_reward = unrealized_return * position_ratio * 0.1 * reward_amplifier
                total_reward = round(total_reward, 3)  # 保留3位小数精度
                
                # 打印持有奖励信息
                print(f"\n=== 持有交易奖励计算 ===")
                print(f"当前持仓比例: {position_ratio*100:.2f}%")
                print(f"未实现收益率: {unrealized_return*100:.4f}%")
                if total_reward >= 0:
                    print(f"持有奖励值: +{total_reward:.3f}\n")
                else:
                    print(f"持有惩罚值: {total_reward:.3f}\n")
        
        # 其他奖励和惩罚
        # 1. 资金利用率奖励
        if self.position > 0:
            utilization_ratio = self.position_value / self.initial_balance
            if 0.3 <= utilization_ratio <= 0.7:  # 合理的资金利用率范围
                utilization_reward = 0.01
                utilization_reward = round(utilization_reward, 3)  # 保留3位小数精度
                total_reward += utilization_reward
                print(f"资金利用率奖励值: +{utilization_reward:.3f} (利用率: {utilization_ratio*100:.1f}%)")
        
        # 2. 资金不足惩罚
        if action == 2 and self.balance < self.fixed_trade_amount:
            insufficient_penalty = -0.2
            insufficient_penalty = round(insufficient_penalty, 3)  # 保留3位小数精度
            total_reward += insufficient_penalty
            print(f"资金不足惩罚值: {insufficient_penalty:.3f}")
        
        # 3. 长时间不活动惩罚
        if self.inaction_steps > 20:
            inaction_penalty = self.reward_config.get('inaction_time_penalty', -0.01) * min(self.inaction_steps / 10, 1.0)
            inaction_penalty = round(inaction_penalty, 3)  # 保留3位小数精度
            total_reward += inaction_penalty
            print(f"不活动惩罚值: {inaction_penalty:.3f} ({self.inaction_steps}步无交易)")
        
        # 最终奖励保留3位小数精度
        total_reward = round(total_reward, 3)
        
        # 打印最终奖励组成详情
        print(f"\n=== 奖励组成详情 ===")
        
        # 检查是否为卖出交易
        is_sell_action = action == 0 and hasattr(self, 'last_actual_profit')
        
        # 卖出交易详情
        if is_sell_action:
            if hasattr(self, 'last_trade_return'):
                # 收益率奖励
                return_reward = self.last_trade_return * reward_amplifier
                if self.last_trade_return > 0 and return_reward < 0.001:
                    return_reward = 0.001
                
                return_reward = round(return_reward, 3)  # 保留3位小数精度
                
                if return_reward >= 0:
                    print(f"- 收益率奖励值: +{return_reward:.3f}")
                else:
                    print(f"- 收益率惩罚值: {return_reward:.3f}")
            
            # 盈利交易奖励
            if self.last_actual_profit > 0:
                profit_base_reward = round(profit_base_reward, 3)  # 保留3位小数精度
                print(f"- 正向利润基础奖励值: +{profit_base_reward:.3f}")
            
            # 趋势方向奖励
            if hasattr(self, 'trend_direction') and self.trend_direction != 0 and hasattr(self, 'last_trade_return') and self.trend_direction * self.last_trade_return > 0:
                trend_reward = self.reward_config.get('trend_follow_reward', 0.1)
                trend_reward = round(trend_reward, 3)  # 保留3位小数精度
                print(f"- 趋势方向奖励值: +{trend_reward:.3f}")
        
        # 输出最终计算得到的奖励值
        if total_reward >= 0:
            print(f"最终奖励值: +{total_reward:.3f}\n")
        else:
            print(f"最终惩罚值: {total_reward:.3f}\n")
            # 当有惩罚值时，输出当前累计奖励值
            # 获取当前环境类型和步数
            env_type = getattr(self, 'env_type', 'training')
            step = getattr(self, 'current_step', 0)
            episode_step = getattr(self, 'episode_step_count', 0)
            
            # 计算当前回合的累计奖励
            cumulative_reward = getattr(self, 'cumulative_reward', 0) + total_reward
            cumulative_reward = round(cumulative_reward, 3)  # 保留3位小数精度
            
            # 输出累计奖励值 - 用步数替代时间
            print(f"【步数: {episode_step}】当前累计奖励值: {cumulative_reward:.3f} (环境: {env_type})\n")
        
        # 强制保证有利润的交易有正向奖励
        if is_sell_action and self.last_actual_profit > 0 and total_reward <= 0:
            total_reward = 0.001  # 最小保底奖励
            print(f"强制修正：有利润交易应有正向奖励，设置为最小奖励值 +0.001")
        
        # 更新累计奖励值
        if not hasattr(self, 'cumulative_reward'):
            self.cumulative_reward = 0
        self.cumulative_reward += total_reward
        self.cumulative_reward = round(self.cumulative_reward, 3)  # 保留3位小数精度
        
        return total_reward
    
    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标到数据中
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的数据
        """
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 简单移动平均线(SMA)
        df['sma7'] = df['close'].rolling(window=7).mean()
        df['sma25'] = df['close'].rolling(window=25).mean()
        
        # 相对强弱指标(RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 填充NaN值
        df = df.bfill().ffill()  # 先向后填充，再向前填充，替代之前的方法
        
        return df
    
    def render(self, mode='human'):
        """
        渲染当前环境状态
        
        参数:
            mode: 渲染模式
        """
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not implemented")
        
        if len(self.trade_history) == 0:
            return
        
        last_record = self.trade_history[-1]
        print(f"Step: {last_record['step']}")
        print(f"Price: {last_record['price']:.2f}")
        print(f"Action: {last_record['action']}")
        print(f"Shares Held: {last_record['shares_held']}")
        print(f"Balance: ${last_record['balance']:.2f}")
        print(f"Portfolio Value: ${last_record['portfolio_value']:.2f}")
        print(f"Reward: {last_record['reward']:.6f}")
        print("-" * 40)
    
    def close(self):
        """
        关闭环境
        """
        pass

    def _calculate_position_size(self, action):
        """基于当前市场状态和账户状态确定仓位大小"""
        # 基础仓位比例 - 更保守的策略
        base_position_ratio = 0.06  # 从0.1降低到0.06，使用更小的仓位
        
        # 获取动态仓位调整的乘数
        dynamic_factor = 1.0  # 默认乘数为1
        
        # 动态调整：根据已损失资金比例减少仓位
        total_assets = self.balance + self.position_value
        loss_ratio = (self.initial_balance - total_assets) / self.initial_balance
        
        # 损失越大，仓位越小 - 加强风险管理
        if loss_ratio > 0.1:  # 损失超过10%就开始调整（从20%降低到10%）
            # 更激进的动态因子减少公式
            dynamic_factor = max(0.3, 1.0 - loss_ratio * 1.5)  # 乘以1.5使降低更快
            
            # 记录日志（降低频率以防刷屏）
            if random.random() < 0.1:  # 10%概率打印日志
                print(f"风险防护：当前损失率 {loss_ratio*100:.1f}%，调整仓位为基础的 {dynamic_factor*100:.1f}%")
        
        # 计算最终仓位比例
        position_ratio = base_position_ratio * dynamic_factor
        
        # 确保单笔交易不超过以下两个限制中的较小值：
        # 1. 初始资金的max_trade_amount（通常为5%）
        # 2. 当前资金的一个合理比例
        max_single_trade_ratio = self.max_trade_amount  # 默认为0.05（5%）
        max_by_initial = self.initial_balance * max_single_trade_ratio  # 初始资金的5%
        
        # 根据总资产减少情况，进一步限制交易金额
        if loss_ratio > 0.2:  # 损失超过20%
            # 改用初始资金的更小比例（如2.5%）
            max_by_initial = self.initial_balance * (max_single_trade_ratio / 2)
            
        # 返回较小值作为最终仓位比例
        max_amount = min(max_by_initial, self.balance * position_ratio)
        final_position_ratio = max_amount / self.balance if self.balance > 0 else 0
        
        # 只在需要输出时计算，减少计算量
        if random.random() < 0.7 or final_position_ratio > 0.1:  # 70%概率打印或者仓位较大时
            print(f"仓位限制: 当前余额={self.balance:.2f}, 初始本金{max_single_trade_ratio*100}%={max_by_initial:.2f}, 最大仓位比例={position_ratio:.2f}, 最终仓位比例={final_position_ratio:.2f}")
        
        return final_position_ratio

    def _track_account_status(self, is_reset=False):
        """跟踪账户状态历史"""
        # 计算当前总资产
        current_portfolio_value = self.balance + self.position_value
        
        # 获取当前步骤的时间戳
        timestamp = self.df.iloc[self.current_step].name
        
        # 计算自上次记录以来的价值变化
        value_change = 0
        value_change_pct = 0
        last_portfolio_value = self.portfolio_value_tracker[-1] if self.portfolio_value_tracker else self.initial_balance
        
        if last_portfolio_value > 0:
            value_change = current_portfolio_value - last_portfolio_value
            value_change_pct = value_change / last_portfolio_value * 100
        
        # 获取当前价格变化百分比
        current_price = self.df.iloc[self.current_step]['close']
        last_price = self.df.iloc[self.current_step-1]['close'] if self.current_step > 0 else current_price
        price_change_pct = (current_price - last_price) / last_price * 100 if last_price > 0 else 0
        
        # 更新账户历史
        step_idx = self.current_step - self.window_size
        if step_idx >= 0:  # 确保我们在有效步骤中
            self.portfolio_value_tracker.append(current_portfolio_value)
            self.balance_tracker.append(self.balance)
            self.position_value_tracker.append(self.position_value)
            self.step_tracker.append(self.current_step)
            
            self.account_history.append({
                'step': step_idx,
                'timestamp': timestamp,
                'portfolio_value': current_portfolio_value,
                'balance': self.balance,
                'position': self.position,
                'position_value': self.position_value,
                'price': current_price,
                'value_change': value_change,
                'value_change_pct': value_change_pct,
                'price_change_pct': price_change_pct
            })
            
            # 每20步（或其他适当间隔）打印一次状态，或者当价值变化超过某个阈值时
            if step_idx % 5 == 0 or abs(value_change_pct) > 2.0:
                print(f"===== 账户状态更新 步骤:{step_idx} =====")
                print(f"时间: {timestamp}")
                print(f"总资产: {current_portfolio_value:.2f} (初始: {self.initial_balance:.2f})")
                
                # 显示资产大幅变化
                if abs(value_change_pct) > 2.0:
                    print(f"⚠️ 资产大幅变化: {value_change:.2f} ({value_change_pct:.2f}%)")
                
                # 显示最近5步的资产变化趋势
                if len(self.portfolio_value_tracker) >= 5:
                    print("最近5步资产变化趋势:")
                    for i in range(max(0, len(self.portfolio_value_tracker)-5), len(self.portfolio_value_tracker)-1):
                        if i+1 < len(self.portfolio_value_tracker) and i-5+len(self.portfolio_value_tracker) >= 0:
                            val1 = self.portfolio_value_tracker[i]
                            val2 = self.portfolio_value_tracker[i+1]
                            step1 = self.step_tracker[i]
                            step2 = self.step_tracker[i+1]
                            val_change = val2 - val1
                            val_change_pct = val_change / val1 * 100 if val1 > 0 else 0
                            price_change_pct = self.account_history[i]['price_change_pct']
                            print(f"  步骤{step1} -> 步骤{step2}: {val1:.2f} -> {val2:.2f} (变化: {val_change:.2f}, {val_change_pct:.2f}%, 价格变化: {price_change_pct:.2f}%)")
                
                print(f"余额: {self.balance:.2f}, 持仓价值: {self.position_value:.2f}")
                print(f"持仓数量: {self.position:.6f}, 价格: {current_price:.4f}")
                print(f"交易次数: {self.trade_count}")
                print(f"最大回撤: {self.max_drawdown*100:.2f}%")
                
            # 如果损失相对于初始资本超过一定阈值，发出额外警告
            loss_ratio = (self.initial_balance - current_portfolio_value) / self.initial_balance
            if loss_ratio > 0.5:  # 损失超过50%
                print(f"===============================")
                print(f"⚠️⚠️ 警告: 资产已下降超过50%! 当前值: {current_portfolio_value:.2f}, 初始值: {self.initial_balance:.2f}")
                print(f"下降比例: {loss_ratio*100:.2f}%")
                
                # 显示最近5笔交易记录
                print("最近5笔交易:")
                for i in range(max(0, len(self.trade_history)-5), len(self.trade_history)):
                    trade = self.trade_history[i]
                    print(f"  行为: {trade['action']}, 价格: {trade['price']:.4f}, 金额: {trade.get('trade_amount', 0.00):.2f}, 时间: {trade['timestamp']}")
    
    def _calculate_market_volatility(self):
        """计算近期市场波动率"""
        if self.current_step < self.window_size + 10:
            return 0.15  # 默认波动率
        
        # 使用过去20个价格计算波动率
        lookback = min(20, self.current_step - self.window_size)
        prices = [self.df.iloc[i]['close'] for i in range(self.current_step - lookback, self.current_step)]
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) > 0:
            volatility = np.std(returns)
            annualized_vol = volatility * np.sqrt(252)  # 年化波动率
            return min(0.5, annualized_vol)  # 限制最大值
        
        return 0.15  # 默认波动率

    def _calculate_portfolio_risk(self):
        """计算投资组合的风险水平"""
        if len(self.drawdown_history) < 10:
            return 0.1  # 默认风险水平
        
        # 计算最近的最大回撤
        recent_history = self.drawdown_history[-20:]
        if len(recent_history) > 0:
            peak = max(recent_history)
            current = recent_history[-1]
            recent_drawdown = (peak - current) / peak if peak > 0 else 0
            
            # 结合最大回撤和资金曲线波动性
            returns = np.diff(recent_history) / recent_history[:-1]
            returns = returns[np.isfinite(returns)]  # 过滤无效值
            
            if len(returns) > 0:
                volatility = np.std(returns)
                # 综合风险评分
                risk_score = 0.7 * recent_drawdown + 0.3 * min(0.5, volatility * 10)
                return min(0.5, risk_score)  # 限制最大风险评分
        
        return 0.1  # 默认风险水平

    def _calculate_buy_amount(self, max_amount):
        """
        计算买入数量，基于当前市场状态和仓位管理策略
        
        参数：
            max_amount: 最大可买入数量
            
        返回：
            实际买入数量
        """
        # 获取当前价格
        current_price = self.df.iloc[self.current_step]['close']
        
        # 如果使用固定交易金额
        if hasattr(self, 'fixed_trade_amount') and self.fixed_trade_amount > 0:
            # 计算固定金额可买入的数量（考虑手续费）
            amount_in_fixed = min(self.fixed_trade_amount / (current_price * (1 + self.transaction_fee)), max_amount)
            print(f"固定交易金额: {self.fixed_trade_amount}，价格: {current_price}，计算数量: {amount_in_fixed}")
            return amount_in_fixed
        
        # 如果使用仓位管理，根据当前资产比例计算买入数量
        if hasattr(self, 'position_sizing') and self.position_sizing:
            # 计算当前资产总值
            total_assets = self.balance + self.position_value
            
            # 计算已有仓位比例
            position_ratio = self.position_value / total_assets if total_assets > 0 else 0
            
            # 根据已有仓位比例调整买入数量
            if position_ratio >= 0.7:  # 已有70%以上资金在仓位上，保守买入
                buy_ratio = 0.1  # 只用10%可用资金买入
            elif position_ratio >= 0.5:  # 已有50%-70%资金在仓位上，适中买入
                buy_ratio = 0.2  # 用20%可用资金买入
            elif position_ratio >= 0.3:  # 已有30%-50%资金在仓位上，较多买入
                buy_ratio = 0.3  # 用30%可用资金买入
            else:  # 仓位较低，激进买入
                buy_ratio = 0.5  # 用50%可用资金买入
                
            # 计算实际买入金额
            buy_amount = self.balance * buy_ratio
            
            # 金额除以价格得到数量
            amount_to_buy = min(max_amount, buy_amount / (current_price * (1 + self.transaction_fee)))
            
            print(f"仓位管理：总资产: {total_assets:.2f}, 仓位比例: {position_ratio:.2f}, 买入比例: {buy_ratio}, 买入金额: {buy_amount:.2f}, 买入数量: {amount_to_buy:.6f}")
            
            return amount_to_buy
        else:
            # 不使用仓位管理，固定使用self.base_position_size比例的资金买入
            if hasattr(self, 'base_position_size'):
                buy_amount = self.balance * self.base_position_size
                amount_to_buy = min(max_amount, buy_amount / (current_price * (1 + self.transaction_fee)))
                print(f"基础仓位：比例: {self.base_position_size}, 买入金额: {buy_amount:.2f}, 买入数量: {amount_to_buy:.6f}")
                return amount_to_buy
            
            # 如果什么都没有设置，就返回最大可买数量
            return max_amount