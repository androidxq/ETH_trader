import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats

class FactorEvaluator:
    """因子评价工具模块

    这个模块用于评估因子的预测能力和实用性，提供了一套完整的因子评价指标体系。

    主要功能：
    1. 计算因子与收益率的相关性（IC值）
    2. 计算因子的稳定性
    3. 评估因子的交易成本（换手率）
    4. 生成因子评价报告

    使用方法示例：
    ```python
    from factor_research.factor_evaluator import FactorEvaluator
    
    # 创建评价器
    evaluator = FactorEvaluator()
    
    # 计算评价指标
    metrics = evaluator.evaluate(
        factor_values=my_factor.get_values(),
        returns=price_returns
    )
    
    # 查看评价结果
    print(metrics)
    ```

    参数说明：
    - factor_values: pd.Series, 因子值序列
    - returns: pd.Series, 对应的收益率序列

    返回指标说明：
    - ic: 信息系数，范围[-1,1]，绝对值越大表示预测能力越强
    - rank_ic: 秩相关系数，对异常值更稳健
    - stability: 稳定性，越接近1表示因子越稳定
    - turnover: 换手率，值越小表示交易成本越低
    """
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, factor_values: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """评价因子
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            
        Returns:
            Dict[str, float]: 评价指标字典
        """
        self.metrics = {
            'ic': self._calculate_ic(factor_values, returns),
            'rank_ic': self._calculate_rank_ic(factor_values, returns),
            'stability': self._calculate_stability(factor_values),
            'turnover': self._calculate_turnover(factor_values),
        }
        return self.metrics
    
    def _calculate_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算信息系数"""
        return factor.corr(returns)
    
    def _calculate_rank_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算秩相关系数"""
        return stats.spearmanr(factor, returns)[0]
    
    def _calculate_stability(self, factor: pd.Series) -> float:
        """计算因子稳定性"""
        return factor.autocorr()
    
    def _calculate_turnover(self, factor: pd.Series) -> float:
        """计算因子换手率"""
        diff = factor.diff().abs()
        return diff.mean() / factor.abs().mean()