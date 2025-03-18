import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SymbolicFactorMiner:
    """符号回归因子挖掘器"""
    
    def __init__(self, 
                 population_size: int = 10000,    # 种群大小
                 generations: int = 500,          # 进化代数
                 tournament_size: int = 20,       # 锦标赛大小
                 stopping_criteria: float = 0.001, # 停止条件
                 early_stopping: int = 50,        # 早停代数
                 const_range: Tuple[float, float] = (-1.0, 1.0)):
        
        # 基础数学运算符
        self.function_set = ['add', 'sub', 'mul', 'div', 
                           'log', 'abs', 'neg', 'sqrt',
                           'sin', 'cos']
        
        # 统计运算符
        self.metric_set = ['mean', 'std', 'max', 'min']
        
        # 时间窗口
        self.windows = [5, 10, 20, 30, 60, 120]
        
        # 初始化符号回归器
        self.regressor = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            const_range=const_range,
            function_set=self.function_set,
            metric='spearman',
            parsimony_coefficient=0.001,  # 控制公式复杂度的惩罚系数
            p_crossover=0.7,             # 交叉概率
            p_subtree_mutation=0.1,      # 子树变异概率
            p_hoist_mutation=0.05,       # 提升变异概率
            p_point_mutation=0.05,       # 点变异概率
            verbose=1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.early_stopping = early_stopping
        self.stopping_criteria = stopping_criteria
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备特征矩阵"""
        features = []
        
        # 1. 基础价格特征
        for col in ['open', 'high', 'low', 'close', 'volume']:
            features.append(data[col].values)
            
        # 2. 价格差异特征
        features.append(data['high'] - data['low'])  # 振幅
        features.append(data['close'] - data['open']) # 实体
        features.append((data['high'] - data['low'])/data['close']) # 相对振幅
        
        # 3. 滚动统计特征
        for window in self.windows:
            for col in ['close', 'volume']:
                for metric in self.metric_set:
                    if metric == 'mean':
                        feat = data[col].rolling(window).mean()
                    elif metric == 'std':
                        feat = data[col].rolling(window).std()
                    elif metric == 'max':
                        feat = data[col].rolling(window).max()
                    elif metric == 'min':
                        feat = data[col].rolling(window).min()
                    features.append(feat.values)
        
        # 4. 动量特征
        for window in self.windows:
            features.append(data['close'].pct_change(window).values)
            
        # 转换为numpy数组并标准化
        X = np.column_stack(features)
        X = np.nan_to_num(X, nan=0)
        return self.scaler.fit_transform(X)
    
    def _prepare_target(self, data: pd.DataFrame, 
                       forward_period: int = 24) -> np.ndarray:
        """准备目标变量（未来收益率）"""
        future_returns = data['close'].shift(-forward_period) / data['close'] - 1
        y = future_returns.values[:-forward_period]
        return y
    
    def mine_factors(self, data: pd.DataFrame, 
                    n_best: int = 3,
                    forward_period: int = 24) -> List[Dict]:
        """挖掘因子"""
        logging.info("开始准备特征...")
        X = self._prepare_features(data)
        y = self._prepare_target(data, forward_period)
        
        # 先移除包含NaN的样本
        valid_idx = ~np.isnan(y)
        y = y[valid_idx]
        X = X[:-forward_period][valid_idx]
        
        # 使用新的验证方法
        X, y = check_X_y(X, y, ensure_2d=True, allow_nd=True)
        
        logging.info(f"数据准备完成，特征矩阵形状: {X.shape}, 目标变量长度: {len(y)}")
        logging.info(f"开始因子挖掘，种群大小={self.regressor.population_size}，进化代数={self.regressor.generations}")
        
        self.regressor.fit(X, y)
        
        # 获取最优程序
        best_programs = []
        for program in self.regressor._programs[-1][:n_best]:
            factor = {
                'expression': str(program),
                'fitness': abs(program.raw_fitness_),
                'complexity': program.length_,
                'program': program
            }
            best_programs.append(factor)
            
        # 按照适应度排序
        best_programs.sort(key=lambda x: x['fitness'], reverse=True)
        
        # 评估最优因子
        for factor in best_programs:
            metrics = self.evaluate_factor(factor, data, forward_period)
            factor.update(metrics)
            
        return best_programs
    
    def evaluate_factor(self, factor: Dict, 
                       data: pd.DataFrame,
                       forward_period: int = 24) -> Dict[str, float]:
        """评估因子表现"""
        X = self._prepare_features(data)
        factor_values = factor['program'].execute(X)
        
        # 计算未来收益
        future_returns = data['close'].shift(-forward_period) / data['close'] - 1
        
        # 确保数据长度匹配
        valid_idx = ~np.isnan(future_returns)
        factor_values = factor_values[:-forward_period][valid_idx[:-forward_period]]
        future_returns = future_returns[:-forward_period][valid_idx[:-forward_period]]
        
        # 计算IC值
        ic = np.corrcoef(factor_values, future_returns)[0,1]
        
        # 计算因子稳定性（使用原始因子值计算稳定性）
        stability = np.corrcoef(factor_values[1:], factor_values[:-1])[0,1]
        
        # 计算因子收益率
        long_threshold = np.percentile(factor_values, 80)
        short_threshold = np.percentile(factor_values, 20)
        
        long_positions = factor_values > long_threshold
        short_positions = factor_values < short_threshold
        
        long_returns = future_returns[long_positions].mean()
        short_returns = -future_returns[short_positions].mean()
        
        return {
            'ic': ic,
            'stability': stability,
            'complexity': factor['complexity'],
            'long_returns': long_returns,
            'short_returns': short_returns
        }