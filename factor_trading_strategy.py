import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import seaborn as sns
from math import *
import operator
import re
from typing import Dict, List, Tuple, Union, Callable

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FactorTradingStrategy:
    """基于因子的交易策略"""

    def __init__(self, 
                 initial_capital: float = 1000.0,
                 transaction_fee_rate: float = 0.04,
                 min_trade_return: float = 0.5,
                 stop_loss: float = 0.5,
                 forward_period: int = 24,
                 long_threshold: float = 0.8,
                 short_threshold: float = 0.2,
                 factor_expression: str = None,
                 trade_direction: str = "只做多"):
        """
        初始化因子交易策略
        
        参数:
            initial_capital (float): 初始资金，默认1000
            transaction_fee_rate (float): 交易手续费率（百分比），默认0.04%
            min_trade_return (float): 最小交易收益要求（百分比），默认0.5%
                                      作为止盈条件：当持仓收益率达到或超过此值时触发平仓
            stop_loss (float): 止损比例（百分比），默认0.5%
                               当持仓亏损达到或超过此值时触发平仓
            forward_period (int): 预测周期（K线数量），默认24
            long_threshold (float): 做多阈值（百分位），默认0.8（前20%）
            short_threshold (float): 做空阈值（百分位），默认0.2（后20%）
            factor_expression (str): 因子表达式，如果为None则需要在子类中实现calculate_factor方法
            trade_direction (str): 交易方向，"只做多"、"只做空"或"多空均做"，默认"只做多"
        """
        self.initial_capital = initial_capital
        self.transaction_fee_rate = transaction_fee_rate / 100.0  # 转换为小数
        self.min_trade_return = min_trade_return / 100.0  # 转换为小数
        self.stop_loss = stop_loss / 100.0  # 转换为小数
        self.forward_period = forward_period
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.factor_expression = factor_expression
        self.trade_direction = trade_direction
        
        # 回测结果
        self.positions = None
        self.equity_curve = None
        self.trades = []
        self.metrics = {}
        self.factor_values = None  # 保存因子值用于UI展示
        self.backtest_result = None  # 保存回测结果数据
        
        # 确保输出目录存在
        self.results_dir = 'trading_results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        参数:
            data (DataFrame): 原始K线数据
            
        返回:
            DataFrame: 添加了特征的数据
        """
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 确保基本列存在
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        # 添加基本特征 X1-X5
        df['X1'] = df['close']
        df['X2'] = df['open']
        df['X3'] = df['high']
        df['X4'] = df['low']
        df['X5'] = df['volume']
        
        # 计算价格变化率 X6-X10
        df['X6'] = df['close'].pct_change(1)
        df['X7'] = df['open'].pct_change(1)
        df['X8'] = df['high'].pct_change(1)
        df['X9'] = df['low'].pct_change(1)
        df['X10'] = df['volume'].pct_change(1)
        
        # 计算移动平均 X11-X15
        for i, window in enumerate([5, 10, 20, 30, 60]):
            df[f'X{11+i}'] = df['close'].rolling(window=window).mean()
        
        # 计算MACD相关指标 X16-X18
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['X16'] = ema12 - ema26  # MACD线
        df['X17'] = df['X16'].ewm(span=9).mean()  # 信号线
        df['X18'] = df['X16'] - df['X17']  # MACD柱状图
        
        # 计算RSI指标 X19-X21
        for i, window in enumerate([6, 12, 24]):
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
            rs = gain / loss
            df[f'X{19+i}'] = 100 - (100 / (1 + rs))
        
        # 计算布林带 X22-X24
        for i, window in enumerate([20, 40, 60]):
            ma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            df[f'X{22+i*3}'] = ma  # 中轨
            df[f'X{23+i*3}'] = ma + 2 * std  # 上轨
            df[f'X{24+i*3}'] = ma - 2 * std  # 下轨
        
        # 计算KDJ指标 X31-X33
        window = 14
        df['highest_high'] = df['high'].rolling(window=window).max()
        df['lowest_low'] = df['low'].rolling(window=window).min()
        df['X31'] = 100 * ((df['close'] - df['lowest_low']) / 
                          (df['highest_high'] - df['lowest_low'] + 1e-10))  # %K
        df['X32'] = df['X31'].rolling(window=3).mean()  # %D
        df['X33'] = 3 * df['X32'] - 2 * df['X31'].rolling(window=3).mean()  # %J
        
        # 计算ATR X34
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['X34'] = tr.rolling(window=14).mean()
        
        # 计算CCI X35
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(window=20).mean()
        md_tp = (tp - ma_tp).abs().rolling(window=20).mean()
        df['X35'] = (tp - ma_tp) / (0.015 * md_tp)
        
        # 计算WILLR X36
        df['X36'] = -100 * ((df['highest_high'] - df['close']) / 
                           (df['highest_high'] - df['lowest_low'] + 1e-10))
        
        # 计算OBV X37
        obv = (df['volume'] * ((df['close'] > df['close'].shift()).astype(int) - 
                              (df['close'] < df['close'].shift()).astype(int))).cumsum()
        df['X37'] = obv
        
        # 计算价格变动率 X38-X40
        for i, period in enumerate([2, 3, 5]):
            df[f'X{38+i}'] = df['close'].pct_change(period)
        
        # 计算不同时间范围的收益率 X41-X45
        for i, period in enumerate([12, 24, 48, 96, 192]):
            df[f'X{41+i}'] = df['close'].pct_change(period)
        
        # 计算波动率指标 X46-X50
        for i, window in enumerate([10, 20, 30, 60, 120]):
            df[f'X{46+i}'] = df['close'].pct_change(1).rolling(window=window).std()
        
        # 计算量价关系指标 X51-X55
        # X51: 价量相关性
        df['X51'] = df['close'].pct_change(1).rolling(window=20).corr(df['volume'].pct_change(1))
        
        # X52: 成交量变动率与价格变动率之比
        df['X52'] = df['X10'] / (df['X6'] + 1e-10)
        
        # X53: 累积成交量变动
        df['X53'] = df['volume'].diff().rolling(window=10).sum()
        
        # X54: 价格变动符号与成交量的乘积
        df['X54'] = np.sign(df['close'].diff()) * df['volume']
        
        # X55: 价格波动与成交量的比率，用于因子表达式中
        df['X55'] = df['close'].pct_change(1).abs().rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)
        
        # 计算更多技术指标 X56-X60
        # X56: 动量指标
        df['X56'] = df['close'] - df['close'].shift(14)
        
        # X57: EMA差值
        df['X57'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        
        # X58: 价格加速度
        df['X58'] = df['close'].diff().diff()
        
        # X59: ROC指标
        df['X59'] = df['close'] / df['close'].shift(10) - 1
        
        # X60: 高低价差与收盘价的比率
        df['X60'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        
        # 清除NaN值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 移除未来数据的计算
        # df['future_return'] = df['close'].shift(-self.forward_period) / df['close'] - 1
        
        return df
    
    def _parse_factor_expression(self, expression: str) -> Callable:
        """
        解析因子表达式
        
        参数:
            expression (str): 因子表达式
            
        返回:
            function: 可执行的函数
        """
        # 定义操作函数
        operations = {
            'add': operator.add,
            'sub': operator.sub,
            'mul': operator.mul,
            'div': lambda a, b: a / (b + 1e-10),  # 避免除以零
            'pow': operator.pow,
            'log': lambda a: np.log(np.abs(a) + 1e-10),
            'sqrt': lambda a: np.sqrt(np.abs(a)),
            'abs': np.abs,
            'max': np.maximum,
            'min': np.minimum,
            'rank': lambda a: pd.Series(a).rank(),
            'delay': lambda a, n: pd.Series(a).shift(int(n)),
            'ts_mean': lambda a, n: pd.Series(a).rolling(window=int(n)).mean(),
            'ts_std': lambda a, n: pd.Series(a).rolling(window=int(n)).std(),
            'ts_min': lambda a, n: pd.Series(a).rolling(window=int(n)).min(),
            'ts_max': lambda a, n: pd.Series(a).rolling(window=int(n)).max(),
            'ts_sum': lambda a, n: pd.Series(a).rolling(window=int(n)).sum(),
            'ts_corr': lambda a, b, n: pd.Series(a).rolling(window=int(n)).corr(pd.Series(b)),
            'ts_cov': lambda a, b, n: pd.Series(a).rolling(window=int(n)).cov(pd.Series(b))
        }
        
        def eval_factor(data):
            # 创建局部变量字典
            local_dict = {f'X{i}': data[f'X{i}'] for i in range(1, 61) if f'X{i}' in data.columns}
            local_dict.update(operations)
            local_dict.update({
                'np': np,
                'pd': pd,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'arcsin': np.arcsin,
                'arccos': np.arccos,
                'arctan': np.arctan,
                'sinh': np.sinh,
                'cosh': np.cosh,
                'tanh': np.tanh,
                'exp': np.exp,
                'log10': np.log10,
                'log2': np.log2
            })
            
            # 替换"X数字"为"X[数字]"以确保安全访问
            safe_expr = expression
            
            # 执行表达式
            try:
                result = eval(safe_expr, {"__builtins__": {}}, local_dict)
                return result
            except Exception as e:
                print(f"表达式计算错误: {e}")
                print(f"表达式: {expression}")
                return pd.Series(np.nan, index=data.index)
        
        return eval_factor
    
    def calculate_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值
        
        参数:
            data (pd.DataFrame): 输入数据
            
        返回:
            pd.Series: 因子值
        """
        if self.factor_expression:
            # 使用表达式计算因子
            factor_func = self._parse_factor_expression(self.factor_expression)
            factor_values = factor_func(data)
            return factor_values
        else:
            # 子类需要实现calculate_factor方法
            raise NotImplementedError("请提供因子表达式或在子类中实现calculate_factor方法")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data (pd.DataFrame): 包含因子和价格数据的DataFrame
            
        返回:
            pd.DataFrame: 添加了signal列的DataFrame
        """
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 计算因子值
        df['factor'] = self.calculate_factor(df)
        
        # 保存因子值以便后续UI使用
        self.factor_values = df['factor']
        
        # 修改为只使用历史数据计算因子排名
        # 使用前100个周期（不包括当前值）的滚动窗口
        df['factor_rank'] = np.nan
        for i in range(100, len(df)):
            # 只使用到当前为止的历史数据计算百分位排名
            historical_data = df['factor'].iloc[i-100:i]
            if not historical_data.isna().all():  # 确保有足够的非NaN数据
                current_value = df['factor'].iloc[i]
                # 计算当前值在历史数据中的百分位排名
                rank_pct = (historical_data < current_value).mean()
                df.loc[df.index[i], 'factor_rank'] = rank_pct
        
        # 根据因子排名生成信号
        # 1: 多头信号，-1: 空头信号，0: 无信号
        df['signal'] = 0
        
        # 根据交易方向设置信号
        if self.trade_direction in ["只做多", "多空均做"]:
            # 当因子排名高于阈值时，做多
            df.loc[df['factor_rank'] > self.long_threshold, 'signal'] = 1
            
        if self.trade_direction in ["只做空", "多空均做"]:
            # 当因子排名低于阈值时，做空
            df.loc[df['factor_rank'] < self.short_threshold, 'signal'] = -1
        
        return df
    
    def backtest(self, data: pd.DataFrame, check_future_data: bool = False) -> pd.DataFrame:
        """
        执行回测
        
        参数:
            data (pd.DataFrame): 输入数据
            check_future_data (bool): 是否检查未来数据使用，默认为False
            
        返回:
            pd.DataFrame: 回测结果
        """
        print(f"准备开始回测，数据量: {len(data)} 条")
        if len(data) > 0:
            print(f"数据时间范围: {data.index[0]} 至 {data.index[-1]}")
        print(f"交易方向: {self.trade_direction}")
        print(f"预测周期: {self.forward_period} 根K线")
        print(f"做多阈值: {self.long_threshold*100:.1f}%")
        print(f"做空阈值: {self.short_threshold*100:.1f}%")
        
        # 如果需要，检查未来数据
        if check_future_data and len(data) > 200:  # 确保有足够数据进行检查
            self.check_for_future_data(data)
        
        # 准备特征和数据
        print("正在准备特征数据...")
        df = self.prepare_features(data)
        print(f"特征准备完成，共计算 {len(df.columns)} 个特征")
        
        # 生成信号
        print("正在生成交易信号...")
        df = self.generate_signals(df)
        print("信号生成完成")
        
        # 初始化回测结果
        print("开始执行回测...")
        capital = self.initial_capital
        equity = []  # 修改：初始化为空列表，只在循环中添加值
        position = 0  # 0: 空仓, 1: 多头, -1: 空头
        shares = 0  # 持仓数量
        entry_price = 0  # 入场价格
        entry_time = None  # 入场时间
        entry_index = 0  # 入场时的索引位置
        trades = []  # 交易记录
        
        # 计算总长度和进度显示间隔
        total_rows = len(df)
        progress_step = max(1, total_rows // 20)  # 显示约20次进度
        
        # 修改：创建回测索引列表，用于保存回测对应的时间点
        backtest_indices = []
        
        # 添加：在第一个回测点记录初始资金
        if total_rows > 100:
            backtest_indices.append(df.index[100])
            equity.append(capital)
        
        for i in range(101, total_rows):  # 从索引101开始，确保已有初始资金记录
            # 显示进度
            if i % progress_step == 0 or i == total_rows - 1:
                progress_pct = ((i - 101) / (total_rows - 101)) * 100
                print(f"回测进度: {i-101}/{total_rows-101} ({progress_pct:.2f}%)")
            
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            current_signal = df['signal'].iloc[i]
            
            # 平仓逻辑
            if position != 0:
                # 计算当前未实现收益率
                unrealized_return = position * (current_price - entry_price) / entry_price
                
                # 当信号反转，或者达到止盈（收益率>=min_trade_return），或者触发止损时平仓
                holding_periods = i - entry_index
                if ((position == 1 and current_signal < 0) or 
                    (position == -1 and current_signal > 0) or 
                    (unrealized_return >= self.min_trade_return) or  # 止盈条件
                    (unrealized_return <= -self.stop_loss)):  # 止损条件
                    
                    # 记录平仓原因（仅用于调试）
                    exit_reason = ""
                    if (position == 1 and current_signal < 0) or (position == -1 and current_signal > 0):
                        exit_reason = "信号反转"
                    elif unrealized_return >= self.min_trade_return:
                        exit_reason = f"止盈触发({unrealized_return*100:.2f}% >= {self.min_trade_return*100:.2f}%)"
                    elif unrealized_return <= -self.stop_loss:
                        exit_reason = f"止损触发({unrealized_return*100:.2f}% <= -{self.stop_loss*100:.2f}%)"
                    
                    # 计算收益
                    exit_price = current_price
                    pnl = position * shares * (exit_price - entry_price)
                    
                    # 计算手续费
                    fee = shares * exit_price * self.transaction_fee_rate
                    net_pnl = pnl - fee
                    
                    # 更新资金
                    capital += net_pnl
                    
                    # 记录交易
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'position': position,
                        'shares': shares,
                        'pnl': pnl,
                        'fee': fee,
                        'net_pnl': net_pnl,
                        'return': net_pnl / (shares * entry_price),
                        'holding_periods': holding_periods,
                        'exit_reason': exit_reason  # 添加平仓原因
                    }
                    trades.append(trade)
                    
                    # 重置仓位
                    position = 0
                    shares = 0
            
            # 开仓 - 移除使用future_return的部分
            if position == 0 and current_signal != 0:
                # 直接根据信号开仓，不再使用future_return判断
                position = current_signal
                entry_time = current_time
                entry_price = current_price
                entry_index = i  # 记录入场时的索引
                shares = capital / entry_price  # 全仓
                fee = shares * entry_price * self.transaction_fee_rate
                shares -= fee / entry_price  # 扣除开仓手续费
            
            # 更新仓位
            df.loc[df.index[i], 'position'] = position
            
            # 更新资金曲线（未实现损益）
            if position != 0:
                unrealized_pnl = position * shares * (current_price - entry_price)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            # 修改：同时保存对应的索引和资金值
            backtest_indices.append(df.index[i])
            equity.append(current_equity)
        
        print(f"回测完成，共产生 {len(trades)} 笔交易")
        if len(trades) > 0:
            print(f"交易时间范围: {trades[0]['entry_time']} 至 {trades[-1]['exit_time']}")
            
            # 统计多空交易次数
            long_trades = sum(1 for t in trades if t['position'] > 0)
            short_trades = sum(1 for t in trades if t['position'] < 0)
            print(f"多头交易: {long_trades}次, 空头交易: {short_trades}次")
            
            # 统计平均持仓周期
            avg_holding = sum(t['holding_periods'] for t in trades) / len(trades)
            print(f"平均持仓周期: {avg_holding:.2f}根K线")
            
            # 统计不同平仓原因的次数
            if 'exit_reason' in trades[0]:
                exit_reasons = {}
                for trade in trades:
                    reason = trade['exit_reason']
                    if reason not in exit_reasons:
                        exit_reasons[reason] = 0
                    exit_reasons[reason] += 1
                
                print("\n平仓原因统计:")
                for reason, count in exit_reasons.items():
                    percentage = (count / len(trades)) * 100
                    print(f"  {reason}: {count}次 ({percentage:.2f}%)")
                
                # 统计各平仓原因的平均收益率
                reason_returns = {}
                for trade in trades:
                    reason = trade['exit_reason']
                    if reason not in reason_returns:
                        reason_returns[reason] = []
                    reason_returns[reason].append(trade['return'] * 100)
                
                print("\n各平仓原因的平均收益率:")
                for reason, returns in reason_returns.items():
                    avg_return = sum(returns) / len(returns)
                    print(f"  {reason}: {avg_return:.2f}%")
        
        # 保存回测结果
        self.positions = df['position']
        
        # 修改：使用收集的回测索引创建资金曲线
        if len(equity) > 0:
            self.equity_curve = pd.Series(equity, index=backtest_indices)
        else:
            # 如果没有交易，创建一个仅包含初始资金的曲线
            self.equity_curve = pd.Series([self.initial_capital], index=[df.index[100]])
            
        self.trades = trades
        self.backtest_result = df  # 保存完整的回测结果数据
        
        # 计算性能指标
        print("计算性能指标...")
        self.calculate_metrics()
        
        return df
    
    def calculate_metrics(self) -> Dict:
        """
        计算性能指标
        
        返回:
            Dict: 性能指标字典
        """
        if self.equity_curve is None:
            raise ValueError("请先执行回测")
        
        # 基本指标
        initial_equity = self.equity_curve.iloc[0]
        final_equity = self.equity_curve.iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        # 计算年化收益率
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days > 0:
            annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        else:
            annual_return = 0
        
        # 计算最大回撤
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # 交易统计
        trade_count = len(self.trades)
        if trade_count > 0:
            win_trades = [t for t in self.trades if t['net_pnl'] > 0]
            win_rate = len(win_trades) / trade_count * 100
            avg_return = sum(t['return'] for t in self.trades) / trade_count * 100
            avg_win = sum(t['return'] for t in win_trades) / len(win_trades) * 100 if win_trades else 0
            avg_loss = sum(t['return'] for t in self.trades if t['net_pnl'] <= 0) / (trade_count - len(win_trades)) * 100 if trade_count > len(win_trades) else 0
            
            # 计算盈亏比
            if avg_loss != 0:
                profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                profit_loss_ratio = float('inf')
                
            # 计算平均持仓周期
            if 'holding_periods' in self.trades[0]:
                avg_holding_periods = sum(t.get('holding_periods', 0) for t in self.trades) / trade_count
            else:
                avg_holding_periods = 0
        else:
            win_rate = 0
            avg_return = 0
            avg_win = 0
            avg_loss = 0
            profit_loss_ratio = 0
            avg_holding_periods = 0
        
        # 计算夏普比率
        daily_returns = self.equity_curve.pct_change().dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'sharpe_ratio': sharpe_ratio,
            'avg_holding_periods': avg_holding_periods
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_results(self, save_path: str = None) -> None:
        """
        绘制回测结果
        
        参数:
            save_path (str): 保存路径，如果为None则显示图表
        """
        if self.equity_curve is None:
            raise ValueError("请先执行回测")
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        plt.figure(figsize=(16, 12))
        
        # 绘制资金曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve, label='总资金', linewidth=2)
        plt.title('因子策略回测结果', fontsize=15)
        plt.xlabel('时间')
        plt.ylabel('资金')
        plt.grid(True)
        plt.legend()
        
        # 添加性能指标文本
        text = (
            f"总收益: {self.metrics['total_return']:.2f}%\n"
            f"年化收益: {self.metrics['annual_return']:.2f}%\n"
            f"最大回撤: {self.metrics['max_drawdown']:.2f}%\n"
            f"交易次数: {self.metrics['trade_count']}\n"
            f"胜率: {self.metrics['win_rate']:.2f}%\n"
            f"平均收益: {self.metrics['avg_return']:.2f}%\n"
            f"盈亏比: {self.metrics['profit_loss_ratio']:.2f}\n"
            f"夏普比率: {self.metrics['sharpe_ratio']:.2f}"
        )
        plt.figtext(0.15, 0.92, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # 绘制交易记录
        plt.subplot(2, 1, 2)
        for trade in self.trades:
            if trade['position'] == 1:  # 多头
                color = 'green' if trade['net_pnl'] > 0 else 'red'
                plt.plot([trade['entry_time'], trade['exit_time']], 
                        [trade['entry_price'], trade['exit_price']], 
                        color=color, linewidth=1.5)
                plt.scatter(trade['entry_time'], trade['entry_price'], color='blue', s=30)
                plt.scatter(trade['exit_time'], trade['exit_price'], color=color, s=30)
            else:  # 空头
                color = 'green' if trade['net_pnl'] > 0 else 'red'
                plt.plot([trade['entry_time'], trade['exit_time']], 
                        [trade['entry_price'], trade['exit_price']], 
                        color=color, linewidth=1.5, linestyle='--')
                plt.scatter(trade['entry_time'], trade['entry_price'], color='orange', s=30)
                plt.scatter(trade['exit_time'], trade['exit_price'], color=color, s=30)
        
        plt.title('交易记录')
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.grid(True)
        
        # 保存或显示
        if save_path:
            try:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                print(f"图表已保存至: {save_path}")
            except Exception as e:
                print(f"保存图表时出错: {e}")
                # 尝试使用不含中文的标题保存
                plt.title('Trading Records')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                print(f"已使用英文标题保存图表至: {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def save_results(self, name: str) -> None:
        """
        保存回测结果
        
        参数:
            name (str): 结果文件名前缀
        """
        if self.equity_curve is None:
            raise ValueError("请先执行回测")
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_prefix = f"{self.results_dir}/{name}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 保存资金曲线
        try:
            self.equity_curve.to_csv(f"{result_prefix}_equity.csv")
            print(f"资金曲线已保存至: {result_prefix}_equity.csv")
        except Exception as e:
            print(f"保存资金曲线时出错: {e}")
        
        # 保存交易记录
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            try:
                trades_df.to_csv(f"{result_prefix}_trades.csv", index=False)
                print(f"交易记录已保存至: {result_prefix}_trades.csv")
            except Exception as e:
                print(f"保存交易记录时出错: {e}")
        
        # 保存性能指标
        try:
            pd.Series(self.metrics).to_csv(f"{result_prefix}_metrics.csv")
            print(f"性能指标已保存至: {result_prefix}_metrics.csv")
        except Exception as e:
            print(f"保存性能指标时出错: {e}")
        
        # 保存图表
        try:
            self.plot_results(f"{result_prefix}_chart.png")
        except Exception as e:
            print(f"保存图表时出错: {e}")
            # 尝试使用备用方法保存图表
            try:
                # 设置简单的英文标题
                plt.figure(figsize=(16, 12))
                plt.plot(self.equity_curve)
                plt.title('Equity Curve')
                plt.savefig(f"{result_prefix}_simple_chart.png")
                print(f"简化图表已保存至: {result_prefix}_simple_chart.png")
            except Exception as e2:
                print(f"保存简化图表也失败: {e2}")
        
        print(f"\n所有结果已保存至: {result_prefix}_*.csv 和 {result_prefix}_chart.png")
    
    def print_summary(self) -> None:
        """打印回测摘要"""
        if not self.metrics:
            raise ValueError("请先执行回测")
        
        print("\n======= 回测摘要 =======")
        print(f"初始资金: {self.initial_capital:.2f} USDT")
        print(f"最终资金: {self.equity_curve.iloc[-1]:.2f} USDT")
        print(f"总收益率: {self.metrics['total_return']:.2f}%")
        print(f"年化收益: {self.metrics['annual_return']:.2f}%")
        print(f"最大回撤: {self.metrics['max_drawdown']:.2f}%")
        print(f"交易次数: {self.metrics['trade_count']}")
        print(f"胜率: {self.metrics['win_rate']:.2f}%")
        print(f"平均收益: {self.metrics['avg_return']:.2f}%")
        print(f"平均盈利: {self.metrics['avg_win']:.2f}%")
        print(f"平均亏损: {self.metrics['avg_loss']:.2f}%")
        print(f"盈亏比: {self.metrics['profit_loss_ratio']:.2f}")
        print(f"夏普比率: {self.metrics['sharpe_ratio']:.2f}")
        print(f"平均持仓周期: {self.metrics.get('avg_holding_periods', 0):.2f}根K线")
        print(f"交易方向: {self.trade_direction}")
        print("=======================")

    def print_english_summary(self) -> None:
        """Print backtest summary in English"""
        if not self.metrics:
            raise ValueError("Please run backtest first")
        
        print("\n======= Backtest Summary =======")
        print(f"Initial Capital: {self.initial_capital:.2f} USDT")
        print(f"Final Capital: {self.equity_curve.iloc[-1]:.2f} USDT")
        print(f"Total Return: {self.metrics['total_return']:.2f}%")
        print(f"Annual Return: {self.metrics['annual_return']:.2f}%")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        print(f"Number of Trades: {self.metrics['trade_count']}")
        print(f"Win Rate: {self.metrics['win_rate']:.2f}%")
        print(f"Average Return: {self.metrics['avg_return']:.2f}%")
        print(f"Average Win: {self.metrics['avg_win']:.2f}%")
        print(f"Average Loss: {self.metrics['avg_loss']:.2f}%")
        print(f"Profit/Loss Ratio: {self.metrics['profit_loss_ratio']:.2f}")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Average Holding Periods: {self.metrics.get('avg_holding_periods', 0):.2f} K-lines")
        print(f"Trade Direction: {self.trade_direction}")
        print("===============================")

    def check_for_future_data(self, data: pd.DataFrame) -> bool:
        """
        检查计算过程中是否使用了未来数据
        
        参数:
            data (pd.DataFrame): 输入数据
            
        返回:
            bool: 如果未发现未来数据使用，返回True；否则返回False
        """
        print("\n===== 未来数据检查 =====")
        future_data_check = True
        
        # 检查1: 确保因子计算只用到当前和过去的数据
        print("1. 检查因子计算...")
        try:
            # 测试：只用一半的数据计算因子值，看结果是否与全量数据计算结果相同
            half_point = len(data) // 2
            half_data = data.iloc[:half_point].copy()
            
            # 准备特征
            half_features = self.prepare_features(half_data)
            full_features = self.prepare_features(data)
            
            # 计算因子值
            half_factor = self.calculate_factor(half_features)
            full_factor = self.calculate_factor(full_features)
            
            # 比较前半部分是否一致
            if half_factor.iloc[:half_point-100].equals(full_factor.iloc[:half_point-100]):
                print("  ✓ 因子计算通过未来数据检查")
            else:
                print("  ✗ 警告：因子计算可能使用了未来数据")
                future_data_check = False
        except Exception as e:
            print(f"  ! 因子计算检查过程中出错: {e}")
        
        # 检查2: 确保信号生成只用到当前和过去的数据
        print("2. 检查信号生成...")
        try:
            # 使用一半数据生成信号
            half_data_with_signals = self.generate_signals(half_features)
            full_data_with_signals = self.generate_signals(full_features)
            
            # 比较前半部分信号是否一致
            half_signals = half_data_with_signals['signal'].iloc[:half_point-100]
            full_signals = full_data_with_signals['signal'].iloc[:half_point-100]
            
            if half_signals.equals(full_signals):
                print("  ✓ 信号生成通过未来数据检查")
            else:
                print("  ✗ 警告：信号生成可能使用了未来数据")
                future_data_check = False
        except Exception as e:
            print(f"  ! 信号生成检查过程中出错: {e}")
        
        print(f"检查结果: {'通过' if future_data_check else '未通过'}")
        print("======================")
        
        return future_data_check 