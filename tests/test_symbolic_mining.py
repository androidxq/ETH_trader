import pandas as pd
from pathlib import Path
import logging
from factor_research.symbolic_miner import SymbolicFactorMiner

def test_symbolic_mining():
    """测试符号回归因子挖掘"""
    # 加载数据
    data_path = Path("d:/pythonProject/ETH_trader/data/kline/ETHUSDT_1h_202401.csv")
    data = pd.read_csv(data_path)
    
    # 创建挖掘器
    miner = SymbolicFactorMiner(
        population_size=10000,
        generations=50,
        early_stopping=50
    )
    
    # 挖掘因子
    factors = miner.mine_factors(data, n_best=3, forward_period=24)
    
    # 打印结果
    for i, factor in enumerate(factors, 1):
        print(f"\n因子 {i}:")
        print(f"表达式: {factor['expression']}")
        print(f"预测能力(IC): {factor['ic']:.4f}")
        print(f"稳定性: {factor['stability']:.4f}")
        print(f"做多收益: {factor['long_returns']:.4f}")
        print(f"做空收益: {factor['short_returns']:.4f}")
        print(f"复杂度: {factor['complexity']}")

if __name__ == "__main__":
    test_symbolic_mining()