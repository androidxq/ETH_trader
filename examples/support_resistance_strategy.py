"""
基于支撑阻力因子的交易策略示例

这个示例展示如何挖掘支撑阻力因子并使用它们构建交易策略。
支撑阻力因子能够识别重要的价格水平，在这些水平上价格可能会反转或突破。
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# 将项目根目录添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto_trader import mine_new_factors, backtest_strategy
from data_processors.binance_processor import load_historical_data
from factor_research.evaluator import calculate_factor_returns
from utils.logger import setup_logger

# 设置日志记录
logger = setup_logger('support_resistance_strategy')

def main():
    """主函数"""
    # 1. 加载数据
    logger.info("加载历史数据...")
    # 从Binance获取ETH/USDT的1小时K线数据
    start_date = datetime.now() - timedelta(days=180)  # 获取最近180天的数据
    df = load_historical_data(
        symbol="ETHUSDT", 
        interval="1h", 
        start_date=start_date,
        save_path="data/eth_1h_support_resistance.csv"
    )
    
    # 2. 数据预处理
    logger.info("数据预处理...")
    # 确保数据时间序列正确
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    # 分割为训练集和测试集
    train_ratio = 0.7
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    logger.info(f"训练集: {train_df.index[0]} 到 {train_df.index[-1]}, 共 {len(train_df)} 条数据")
    logger.info(f"测试集: {test_df.index[0]} 到 {test_df.index[-1]}, 共 {len(test_df)} 条数据")
    
    # 3. 挖掘支撑阻力因子
    logger.info("开始挖掘支撑阻力因子...")
    support_resistance_factors = mine_new_factors(
        data=train_df,
        factor_type="支撑阻力因子",
        forward_period=24,  # 24小时预测期
        n_best=3  # 返回前3个最佳因子
    )
    
    # 4. 构建组合因子信号
    logger.info("构建组合因子信号...")
    # 在测试集上计算因子值
    factor_values = []
    factor_weights = []
    
    # 根据训练集上的表现分配权重
    total_ic = sum(abs(factor['ic']) for factor in support_resistance_factors)
    
    for factor in support_resistance_factors:
        # 计算因子权重（根据IC值的比例）
        weight = abs(factor['ic']) / total_ic if total_ic > 0 else 1.0 / len(support_resistance_factors)
        factor_weights.append(weight)
        
        # 解析因子表达式并计算值
        expression = factor['expression']
        logger.info(f"计算因子: {expression}, 权重: {weight:.4f}")
        
        # 在此示例中我们简化为直接使用函数的计算结果
        # 在实际应用中应该解析表达式并计算
        from factor_research.symbolic_miner import SymbolicFactorMiner
        miner = SymbolicFactorMiner(function_set=["add", "sub", "mul", "div", "log", "sqrt", "square", "pow", "exp"])
        factor_value = miner.evaluate_expression(test_df, expression, factor_type="支撑阻力因子")
        factor_values.append(factor_value)
    
    # 组合多个因子信号
    combined_signal = np.zeros(len(test_df))
    for i, factor_value in enumerate(factor_values):
        # 标准化因子值
        normalized_factor = (factor_value - np.nanmean(factor_value)) / np.nanstd(factor_value)
        combined_signal += normalized_factor * factor_weights[i]
    
    # 5. 设置交易信号阈值
    logger.info("设置交易信号阈值并生成交易信号...")
    buy_threshold = 1.0  # 当组合信号 > 1.0 标准差时买入
    sell_threshold = -1.0  # 当组合信号 < -1.0 标准差时卖出
    
    # 生成交易信号
    signals = pd.Series(0, index=test_df.index)
    signals[combined_signal > buy_threshold] = 1  # 买入信号
    signals[combined_signal < sell_threshold] = -1  # 卖出信号
    
    # 6. 回测策略
    logger.info("开始回测策略...")
    backtest_results = backtest_strategy(
        data=test_df,
        signals=signals,
        initial_capital=10000,
        position_size=1.0,
        fee_rate=0.001,  # 0.1% 交易费率
        verbose=True
    )
    
    # 7. 分析回测结果
    logger.info("分析回测结果...")
    print("\n=== 回测结果摘要 ===")
    print(f"初始资本: ${backtest_results['initial_capital']:.2f}")
    print(f"最终资本: ${backtest_results['final_capital']:.2f}")
    print(f"总收益率: {backtest_results['total_return']*100:.2f}%")
    print(f"年化收益率: {backtest_results['annual_return']*100:.2f}%")
    print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {backtest_results['max_drawdown']*100:.2f}%")
    print(f"胜率: {backtest_results['win_rate']*100:.2f}%")
    print(f"总交易次数: {backtest_results['total_trades']}")
    
    # 8. 绘制回测图表
    plt.figure(figsize=(14, 10))
    
    # 价格和权益曲线
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(test_df.index, test_df['close'], 'b-', alpha=0.3, label='ETH价格')
    ax1.set_ylabel('ETH价格 (USDT)')
    ax1.set_title('支撑阻力因子策略回测')
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(backtest_results['equity_curve'].index, 
                 backtest_results['equity_curve'], 'g-', label='权益曲线')
    ax1_twin.set_ylabel('账户权益 (USDT)')
    ax1_twin.legend(loc='upper right')
    
    # 交易信号
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(test_df.index, combined_signal, 'b-', label='组合因子信号')
    ax2.axhline(y=buy_threshold, color='g', linestyle='--', label='买入阈值')
    ax2.axhline(y=sell_threshold, color='r', linestyle='--', label='卖出阈值')
    ax2.fill_between(test_df.index, buy_threshold, combined_signal, 
                    where=combined_signal > buy_threshold, color='g', alpha=0.3)
    ax2.fill_between(test_df.index, sell_threshold, combined_signal, 
                    where=combined_signal < sell_threshold, color='r', alpha=0.3)
    ax2.set_ylabel('信号强度')
    ax2.legend()
    
    # 买入卖出点
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(test_df.index, test_df['close'], 'k-', alpha=0.3)
    
    # 标记买入点
    buy_points = signals[signals == 1].index
    sell_points = signals[signals == -1].index
    
    for idx in buy_points:
        if idx in test_df.index:
            ax3.scatter(idx, test_df.loc[idx, 'close'], marker='^', color='g', s=100)
    
    for idx in sell_points:
        if idx in test_df.index:
            ax3.scatter(idx, test_df.loc[idx, 'close'], marker='v', color='r', s=100)
    
    ax3.set_ylabel('ETH价格 (USDT)')
    ax3.set_xlabel('日期')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('results/support_resistance_strategy_backtest.png')
    logger.info("回测图表已保存到 results/support_resistance_strategy_backtest.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main() 