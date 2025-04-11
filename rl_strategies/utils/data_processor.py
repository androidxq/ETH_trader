"""
数据处理工具

包含用于处理交易数据的工具函数和类
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class DataProcessor:
    """数据处理类"""
    
    @staticmethod
    def add_technical_indicators(
        data: pd.DataFrame, 
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        添加技术指标到数据中
        
        参数:
            data: 原始OHLCV数据
            indicators: 要添加的指标列表，为None时添加全部指标
            
        返回:
            添加了技术指标的数据
        """
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 如果没有指定指标，则添加全部指标
        if indicators is None:
            indicators = [
                'sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'obv', 'roc'
            ]
        
        # 确保数据中包含必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"数据中缺少必要的列: {col}")
        
        # 添加指标
        if 'sma' in indicators:
            DataProcessor._add_sma(df)
        
        if 'ema' in indicators:
            DataProcessor._add_ema(df)
        
        if 'rsi' in indicators:
            DataProcessor._add_rsi(df)
        
        if 'macd' in indicators:
            DataProcessor._add_macd(df)
        
        if 'bbands' in indicators:
            DataProcessor._add_bbands(df)
        
        if 'atr' in indicators:
            DataProcessor._add_atr(df)
        
        if 'obv' in indicators:
            DataProcessor._add_obv(df)
        
        if 'roc' in indicators:
            DataProcessor._add_roc(df)
        
        # 填充NaN值
        # df.fillna(method='bfill', inplace=True)
        # df.fillna(method='ffill', inplace=True)
        
        # 使用推荐的方法替代
        df = df.bfill().ffill()
        
        # 去除仍然包含NaN的行
        df = df.dropna()
        
        return df
    
    @staticmethod
    def normalize_data(
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        归一化数据
        
        参数:
            data: 原始数据
            columns: 要归一化的列，为None时归一化所有数值列
            method: 归一化方法，'minmax'或'zscore'
            
        返回:
            归一化后的数据
        """
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 如果没有指定列，则选择所有数值列
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        # 进行归一化
        if method == 'minmax':
            for col in columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        return df
    
    @staticmethod
    def split_data(
        data: pd.DataFrame, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = False,
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        拆分数据为训练集、验证集和测试集
        
        参数:
            data: 原始数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱数据
            seed: 随机种子
            
        返回:
            训练集、验证集和测试集
        """
        # 确保比例合法
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("训练集、验证集和测试集的比例之和必须为1")
        
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 如果需要打乱数据
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            df = df.sample(frac=1).reset_index(drop=True)
        
        # 计算拆分点
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # 拆分数据
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        return train_data, val_data, test_data
    
    @staticmethod
    def _add_sma(df: pd.DataFrame) -> None:
        """添加简单移动平均线"""
        df['sma7'] = df['close'].rolling(window=7).mean()
        df['sma25'] = df['close'].rolling(window=25).mean()
        df['sma99'] = df['close'].rolling(window=99).mean()
    
    @staticmethod
    def _add_ema(df: pd.DataFrame) -> None:
        """添加指数移动平均线"""
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    @staticmethod
    def _add_rsi(df: pd.DataFrame) -> None:
        """添加相对强弱指标"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    @staticmethod
    def _add_macd(df: pd.DataFrame) -> None:
        """添加MACD指标"""
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    @staticmethod
    def _add_bbands(df: pd.DataFrame) -> None:
        """添加布林带指标"""
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    @staticmethod
    def _add_atr(df: pd.DataFrame) -> None:
        """添加平均真实范围指标"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
    
    @staticmethod
    def _add_obv(df: pd.DataFrame) -> None:
        """添加能量潮指标"""
        obv = np.zeros(len(df))
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        df['obv'] = obv
    
    @staticmethod
    def _add_roc(df: pd.DataFrame) -> None:
        """添加变动率指标"""
        df['roc'] = df['close'].pct_change(periods=12) * 100 