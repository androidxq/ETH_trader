import pandas as pd
import numpy as np
from typing import List, Tuple

class OptimalPointsFinder:
    """最优买卖点标注工具

    这个模块用于在历史数据中自动标注最优的买入和卖出点，
    通过分析局部最高点和最低点，结合未来收益来确定。

    使用方法：
    from factor_research.optimal_points import OptimalPointsFinder

    # 创建标注工具
    finder = OptimalPointsFinder(lookback_window=20, profit_threshold=0.02)

    # 寻找最优点
    buy_points, sell_points = finder.find_optimal_points(data)

    参数说明：
    - lookback_window: int, 向前后查找的窗口大小
    - profit_threshold: float, 最小收益率阈值
    - data: pd.DataFrame, 包含OHLCV数据的DataFrame

    返回值说明：
    - buy_points: List[int], 买入点索引列表
    - sell_points: List[int], 卖出点索引列表
    """
    def __init__(self, lookback_window: int = 20, profit_threshold: float = 0.02):
        self.lookback_window = lookback_window
        self.profit_threshold = profit_threshold
        
    def find_optimal_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """找出最优买卖点
        
        Args:
            data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close']
            
        Returns:
            Tuple[List[int], List[int]]: 买入点和卖出点的索引列表
        """
        prices = data['close'].values
        buy_points = []
        sell_points = []
        
        for i in range(self.lookback_window, len(prices)-self.lookback_window):
            # 寻找局部最低点（买入点）
            if self._is_local_minimum(prices, i):
                future_profit = self._calculate_future_profit(prices, i)
                if future_profit > self.profit_threshold:
                    buy_points.append(i)
            
            # 寻找局部最高点（卖出点）
            if self._is_local_maximum(prices, i):
                past_profit = self._calculate_past_profit(prices, i)
                if past_profit > self.profit_threshold:
                    sell_points.append(i)
        
        return buy_points, sell_points
    
    def _is_local_minimum(self, prices: np.ndarray, index: int) -> bool:
        """判断是否为局部最低点"""
        window = prices[index-self.lookback_window:index+self.lookback_window+1]
        return prices[index] == min(window)
    
    def _is_local_maximum(self, prices: np.ndarray, index: int) -> bool:
        """判断是否为局部最高点"""
        window = prices[index-self.lookback_window:index+self.lookback_window+1]
        return prices[index] == max(window)
    
    def _calculate_future_profit(self, prices: np.ndarray, index: int) -> float:
        """计算未来可能的最大收益"""
        future_window = prices[index:index+self.lookback_window]
        return (max(future_window) - prices[index]) / prices[index]
    
    def _calculate_past_profit(self, prices: np.ndarray, index: int) -> float:
        """计算过去的最大收益"""
        past_window = prices[index-self.lookback_window:index]
        return (prices[index] - min(past_window)) / min(past_window)