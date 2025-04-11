import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys
from glob import glob
from factor_trading_strategy import FactorTradingStrategy

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 尝试导入项目的数据加载模块
try:
    from factor_research.data_loader import load_data_files
    has_data_loader = True
except ImportError:
    has_data_loader = False

# 从挖掘报告中获取的因子1表达式
FACTOR_EXPRESSION = "div(X10, log(X55))"  # 用户提供的因子表达式

# 从因子挖掘报告中获取的参数
FORWARD_PERIOD = 48        # 预测周期（K线数量）
TRANSACTION_FEE = 0.04     # 交易手续费率（%）
MIN_TRADE_RETURN = 0.05    # 最小交易收益（%）
INITIAL_CAPITAL = 1000.0   # 初始资金（USDT）

def custom_load_data_files(data_pattern):
    """自定义数据加载函数，当项目数据加载模块不可用时使用"""
    all_files = glob(data_pattern)
    if not all_files:
        print(f"未找到匹配 {data_pattern} 的数据文件")
        return pd.DataFrame()
    
    print(f"找到 {len(all_files)} 个数据文件")
    
    # 加载所有匹配的文件
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
            print(f"加载文件: {file}")
        except Exception as e:
            print(f"加载文件 {file} 失败: {str(e)}")
    
    if not dfs:
        return pd.DataFrame()
    
    # 合并数据
    data = pd.concat(dfs)
    
    # 确保数据已排序并去重
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp').drop_duplicates()
    
    return data

def main():
    """主函数：执行因子策略回测"""
    print("======= 因子交易策略回测 =======")
    print(f"使用因子表达式: {FACTOR_EXPRESSION}")
    print(f"预测周期: {FORWARD_PERIOD}根K线")
    print(f"交易手续费率: {TRANSACTION_FEE}%")
    print(f"最小交易收益: {MIN_TRADE_RETURN}%")
    print(f"初始资金: {INITIAL_CAPITAL} USDT")
    print("================================")
    
    # 数据加载
    symbol = "ETHUSDT"
    interval = "5m"
    data_pattern = f"data/kline/{symbol}_{interval}_*.csv"
    
    # 尝试使用项目的数据加载模块
    if has_data_loader:
        print("使用项目内置数据加载器...")
        data = load_data_files(data_pattern)
    else:
        print("使用自定义数据加载函数...")
        data = custom_load_data_files(data_pattern)
    
    # 检查数据加载结果
    if data.empty:
        print(f"无法加载数据，请检查 {data_pattern} 路径是否正确")
        # 列出可用的数据文件
        kline_dir = "data/kline"
        if os.path.exists(kline_dir):
            print(f"\n可用的数据文件:")
            for file in os.listdir(kline_dir):
                if file.endswith(".csv"):
                    print(f" - {file}")
        return
    
    # 数据预处理
    print(f"数据加载成功，共 {len(data)} 行")
    
    # 如果数据量太大，可以只用近期数据
    if len(data) > 20000:
        print(f"数据量较大，只使用最近的20000条记录进行回测")
        data = data.sort_values('timestamp').tail(20000)
    
    # 确保数据已排序
    data = data.sort_values('timestamp')
    
    # 检查数据是否包含必要的列
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"错误: 数据文件缺少必要的列: {missing_columns}")
        return
    
    # 设置timestamp为索引
    if 'timestamp' in data.columns and not data.index.name == 'timestamp':
        data.set_index('timestamp', inplace=True)
    
    print(f"数据时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 创建策略实例
    strategy = FactorTradingStrategy(
        initial_capital=INITIAL_CAPITAL,
        transaction_fee_rate=TRANSACTION_FEE,
        min_trade_return=MIN_TRADE_RETURN,
        forward_period=FORWARD_PERIOD,
        long_threshold=0.8,  # 做多阈值（前20%）
        short_threshold=0.2,  # 做空阈值（后20%）
        factor_expression=FACTOR_EXPRESSION
    )
    
    # 执行回测
    signals = strategy.backtest(data)
    
    # 打印回测摘要
    try:
        # 尝试打印中文摘要
        strategy.print_summary()
    except Exception as e:
        print(f"打印中文摘要时出错: {e}")
    
    # 再打印英文摘要，确保至少有一个能正常显示
    strategy.print_english_summary()
    
    # 保存回测结果
    strategy.save_results("eth_factor1_strategy")
    
    print("\n回测完成！详细回测结果已保存到trading_results目录。")

if __name__ == "__main__":
    main() 