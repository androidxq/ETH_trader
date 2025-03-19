import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array
import logging
import time  # 添加time模块

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SymbolicFactorMiner:
    """符号回归因子挖掘器"""
    
    def __init__(self, 
                 population_size: int = 10000,    # 种群大小
                 generations: int = 500,          # 进化代数
                 tournament_size: int = 20,       # 锦标赛大小
                 stopping_criteria: float = 0.0,  # 停止条件设为0，禁用早停
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
            parsimony_coefficient=0.001,  # 控制公式复杂度的惩罚系数
            p_crossover=0.7,             # 交叉概率
            p_subtree_mutation=0.1,      # 子树变异概率
            p_hoist_mutation=0.05,       # 提升变异概率
            p_point_mutation=0.05,       # 点变异概率
            verbose=1,
            stopping_criteria=stopping_criteria,  # 使用传入的 stopping_criteria
            random_state=42             # 保持固定种子不变
        )
        
        self.scaler = StandardScaler()
        self.early_stopping = early_stopping
        self.stopping_criteria = stopping_criteria
        
        # 添加演化历史记录
        self.evolution_history = []
        
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
        
        # 打印调试信息
        print("\n============= 遗传算法训练参数 =============")
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"预测周期: {forward_period}")
        print(f"种群大小: {self.regressor.population_size}")
        print(f"进化代数: {self.regressor.generations}")
        print(f"锦标赛大小: {self.regressor.tournament_size}")
        print(f"早停代数: {self.early_stopping}")
        print(f"停止条件阈值: {self.stopping_criteria}")
        print("=============================================\n")
        
        # 监控变量初始化
        self.evolution_history = []
        start_time = time.time()
        
        # 保存原始回调函数
        original_callback = getattr(self.regressor, '_callback', None)
        
        # 定义增强的回调函数
        def enhanced_callback(gp, generation):
            """增强的回调函数，记录每代信息"""
            if original_callback:
                original_callback(gp, generation)
            
            # 记录最佳程序
            if hasattr(gp, '_best_program'):
                best_program = gp._best_program
                best_fitness = abs(best_program.raw_fitness_)  # 使用绝对值
                best_expr = str(best_program)
                
                elapsed = time.time() - start_time
                
                # 记录信息
                gen_info = {
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'best_expr': best_expr,
                    'elapsed_time': elapsed
                }
                self.evolution_history.append(gen_info)
                
                # 打印信息
                print(f"\n【调试】第{generation}代完成:")
                print(f"  最佳适应度: {best_fitness}")
                print(f"  最佳表达式: {best_expr}")
                print(f"  耗时: {elapsed:.2f}秒")
                
                # 如果是第0代，检查stopping_criteria设置
                if generation == 0:
                    print(f"\n【调试】stopping_criteria设置值: {getattr(gp, 'stopping_criteria', '未找到')}")
                    print(f"  最佳适应度 {best_fitness} 与停止条件的关系: {'大于' if best_fitness > gp.stopping_criteria else '小于等于'} 停止条件")
                    if best_fitness > gp.stopping_criteria and gp.stopping_criteria > 0:
                        print(f"  【警告】可能触发早停! 第0代适应度({best_fitness})已经超过停止条件({gp.stopping_criteria})")
                
                # 检查是否只有第0代
                if generation == 0 and hasattr(gp, '_iterations') and gp._iterations <= 1:
                    print("\n【警告】可能触发了早停! 只执行了第0代!")
                    print("  请检查以下可能原因:")
                    print("  1. 是否设置了过低的stopping_criteria")
                    print("  2. 初始种群是否包含了特别优秀的个体")
                    
                # 当演化继续到第10代时，确认早停已禁用
                if generation == 10:
                    print("\n【确认】演化已经成功进行了10代，早停机制未触发!")
                    
                # 每10代输出一次进度
                if generation > 0 and generation % 10 == 0:
                    print(f"\n=== 演化进度: 已完成{generation}/{gp.generations}代 ===")
                    if len(self.evolution_history) >= 2:
                        first_gen = self.evolution_history[0]
                        current_gen = self.evolution_history[-1]
                        if first_gen['best_expr'] == current_gen['best_expr']:
                            print("【注意】当前最佳表达式与第0代相同，可能算法陷入局部最优")
                        improvement = current_gen['best_fitness'] - first_gen['best_fitness']
                        print(f"  与第0代相比，适应度改善: {improvement:.6f}")
        
        # 替换回调函数
        self.regressor._callback = lambda gp, generation: enhanced_callback(gp, generation)
        
        # 保存原始的fit方法
        original_fit = self.regressor.fit
        
        # 定义增强的fit方法
        def monitored_fit(X, y, *args, **kwargs):
            print("\n【调试】开始执行fit方法...")
            
            # 检查stopping_criteria属性
            print(f"【调试】stopping_criteria设置为: {getattr(self.regressor, 'stopping_criteria', '未设置')}")
            
            # 拦截_run方法以检测早停
            original_run = getattr(self.regressor, '_run', None)
            
            if original_run:
                def monitored_run(X, y, *args, **kwargs):
                    print("\n【调试】开始执行_run方法...")
                    result = original_run(X, y, *args, **kwargs)
                    print(f"\n【调试】_run方法完成，返回结果类型: {type(result)}")
                    
                    # 检查迭代次数
                    if hasattr(self.regressor, '_iterations'):
                        print(f"【调试】迭代结束后的_iterations值: {self.regressor._iterations}")
                    
                    # 检查最佳程序
                    if hasattr(self.regressor, '_best_program'):
                        print(f"【调试】最佳程序表达式: {str(self.regressor._best_program)}")
                        print(f"【调试】最佳程序适应度: {self.regressor._best_program.raw_fitness_}")
                    
                    return result
                
                # 替换_run方法
                self.regressor._run = monitored_run
            
            # 执行原始fit方法
            result = original_fit(X, y, *args, **kwargs)
            
            print(f"\n【调试】fit方法执行完成")
            if hasattr(self.regressor, '_iterations'):
                print(f"  实际执行迭代次数: {self.regressor._iterations}")
                
                # 添加这一行来判断是否触发了早停
                if self.regressor._iterations <= 1:
                    print("\n【警告】遗传算法提前终止!")
                    print("  只执行了一代就停止，正在检查原因...")
                    
                    # 检查可能的早停原因
                    if hasattr(self.regressor, '_best_program'):
                        fitness = abs(self.regressor._best_program.raw_fitness_)
                        if fitness >= self.stopping_criteria:
                            print(f"  【确认】触发了适应度早停条件: 最佳适应度({fitness}) >= 停止阈值({self.stopping_criteria})")
                        else:
                            print(f"  【排除】适应度早停条件未触发: 最佳适应度({fitness}) < 停止阈值({self.stopping_criteria})")
                    
                    # 检查固定种子问题
                    print(f"  【排查】随机种子设置为: {getattr(self.regressor, 'random_state', '未知')}")
                    print(f"  【建议】尝试修改random_state=None以使用不同随机种子")
            
            # 恢复原始_run方法
            if original_run:
                self.regressor._run = original_run
                
            print(f"  总耗时: {time.time() - start_time:.2f}秒")
            
            return result
        
        # 替换fit方法
        self.regressor.fit = lambda X, y, *args, **kwargs: monitored_fit(X, y, *args, **kwargs)
        
        # 训练模型
        try:
            self.regressor.fit(X, y)
            
            # 训练后检查
            print("\n============= 训练完成信息 =============")
            if hasattr(self.regressor, '_iterations'):
                print(f"执行代数: {self.regressor._iterations}/{self.regressor.generations}")
                
                if self.regressor._iterations < self.regressor.generations:
                    print(f"\n【警告】实际执行代数({self.regressor._iterations})小于设定代数({self.regressor.generations})")
                    print("可能原因:")
                    print("1. 提前达到了stopping_criteria")
                    print("2. 初始种群已包含最优解")
                    print("3. 算法内部逻辑提前终止")
            
            # 显示演化历史统计
            if self.evolution_history:
                first_gen = self.evolution_history[0]
                last_gen = self.evolution_history[-1]
                
                print("\n演化历史:")
                print(f"第一代 - 最佳适应度: {first_gen['best_fitness']}, 表达式: {first_gen['best_expr']}")
                print(f"最后代 - 最佳适应度: {last_gen['best_fitness']}, 表达式: {last_gen['best_expr']}")
                
                if first_gen['best_expr'] == last_gen['best_expr'] and len(self.evolution_history) > 1:
                    print("\n【关键发现】第一代和最后一代的最佳表达式相同!")
                    print("这表明算法可能在第一代就找到了最优解，或者后续进化过程没有产生更好的解。")
        
        except Exception as e:
            logging.error(f"遗传算法训练出错: {str(e)}")
            raise
        finally:
            # 恢复原始方法
            self.regressor.fit = original_fit
            self.regressor._callback = original_callback
        
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