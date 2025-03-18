from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseFactor(ABC):
    """因子基类模块

    这个模块定义了所有因子的基础接口，是开发新因子的模板类。
    所有自定义因子都应该继承这个基类，并实现calculate方法。

    主要功能：
    1. 提供因子计算的标准接口
    2. 管理因子参数
    3. 存储因子计算结果

    使用方法示例：
    ```python
    from factor_research.base_factor import BaseFactor
    
    class MomentumFactor(BaseFactor):
        def __init__(self, window: int = 20):
            super().__init__('momentum')
            self.set_params(window=window)
            
        def calculate(self, data: pd.DataFrame) -> pd.Series:
            window = self.params['window']
            returns = data['close'].pct_change(window)
            self.factor_values = returns
            return returns
    ```
    参数说明：
    - name: str, 因子名称
    - data: pd.DataFrame, 包含OHLCV数据的DataFrame
    - params: Dict, 因子计算参数
    
    方法说明：
    - calculate: 实现因子计算逻辑
    - set_params: 设置因子参数
    - get_values: 获取计算后的因子值
    """
    def __init__(self, name: str):
        self.name = name
        self.params: Dict[str, Any] = {}
        self.factor_values: pd.Series = None
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass
    
    def set_params(self, **kwargs):
        """设置因子参数"""
        self.params.update(kwargs)
        
    def get_values(self) -> pd.Series:
        """获取因子值"""
        return self.factor_values