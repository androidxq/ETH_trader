import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from gplearn.genetic import SymbolicRegressor, _Program
from gplearn.functions import make_function, _Function
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array
import logging
import time  # 添加time模块
import sys
import os  # 添加os模块获取进程ID

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加进程ID的打印函数
def print_with_pid(*args, **kwargs):
    """带进程ID的打印函数"""
    pid = os.getpid()
    prefix = f"[PID-{pid}] "
    
    # 如果是文本字符串，直接添加前缀
    if args and isinstance(args[0], str):
        new_args = (prefix + args[0],) + args[1:]
        print(*new_args, **kwargs)
    else:
        # 先打印前缀，再打印原始内容
        print(prefix, end="")
        print(*args, **kwargs)

class SymbolicFactorMiner:
    """符号回归因子挖掘器"""
    
    def __init__(self, 
                 population_size: int = 10000,    # 种群大小
                 generations: int = 500,          # 进化代数
                 tournament_size: int = 20,       # 锦标赛大小
                 stopping_criteria: float = 0.0,  # 停止条件设为0，禁用早停
                 early_stopping: int = 50,        # 早停代数
                 const_range: Tuple[float, float] = (-1.0, 1.0),
                 p_crossover: float = 0.7,        # 交叉概率
                 p_subtree_mutation: float = 0.1, # 子树变异概率
                 p_hoist_mutation: float = 0.05,  # 提升变异概率
                 p_point_mutation: float = 0.05,  # 点变异概率
                 parsimony_coefficient: float = 0.001, # 复杂度惩罚系数
                 init_depth: Tuple[int, int] = (2, 6), # 初始树深度范围
                 random_state = None):            # 随机种子
        
        # 初始化函数集，使用内置的保护函数
        # 注意: gplearn默认提供保护版本的数学函数，会自动处理无效值:
        # - 'div' 是保护除法 (除以接近0的值返回1)
        # - 'sqrt' 是保护平方根 (负数取绝对值)
        # - 'log' 是保护对数 (负数取绝对值)
        self.function_set = ['add', 'sub', 'mul', 'div',
                             'sqrt', 'log', 'abs', 'neg', 
                             'sin', 'cos']
        
        # 统计运算符
        self.metric_set = ['mean', 'std', 'max', 'min']
        
        # 时间窗口
        self.windows = [5, 10, 20, 30, 60, 120]
        
        # 保存所有参数
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.early_stopping = early_stopping
        self.const_range = const_range
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.parsimony_coefficient = parsimony_coefficient
        self.init_depth = init_depth
        self.random_state = random_state if random_state is not None else int(time.time())
        
        # 使用原生的SymbolicRegressor
        self.regressor = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            const_range=const_range,
            function_set=self.function_set,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            verbose=1,
            stopping_criteria=stopping_criteria,
            init_depth=init_depth,
            random_state=self.random_state
            # 移除feature_names参数，将在运行时动态设置
        )
        
        self.scaler = StandardScaler()
        
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
        
        # 动态设置特征名称
        n_features = X.shape[1]
        self.regressor.feature_names = ['X' + str(i) for i in range(n_features)]
        
        # 打印调试信息
        print_with_pid("\n============= 遗传算法训练参数 =============")
        print_with_pid(f"特征数量: {X.shape[1]}")
        print_with_pid(f"样本数量: {X.shape[0]}")
        print_with_pid(f"预测周期: {forward_period}")
        print_with_pid(f"种群大小: {self.regressor.population_size}")
        print_with_pid(f"进化代数: {self.regressor.generations}")
        print_with_pid(f"锦标赛大小: {self.regressor.tournament_size}")
        print_with_pid(f"早停代数: {self.early_stopping}")
        print_with_pid(f"停止条件阈值: {self.stopping_criteria}")
        print_with_pid("=============================================\n")
        
        # 显示指标解释
        self._explain_metrics()
        
        # 在使用原生的gplearn报告器之前先翻译表头
        print_with_pid("\n【提示】表头含义:")
        print_with_pid("  Gen = 代数       Length = 平均长度      Fitness = 平均适应度")
        print_with_pid("  最右侧Length = 最佳长度    最右侧Fitness = 最佳适应度")
        print_with_pid("  OOB Fitness = 交叉验证评分    Time Left = 预计剩余时间")
        print_with_pid("====================================================\n")
        
        # 监控变量初始化
        self.evolution_history = []
        start_time = time.time()
        
        # 保存原始回调函数
        original_callback = getattr(self.regressor, '_callback', None)
        
        # 定义增强的回调函数
        def enhanced_callback(gp, generation):
            """增强的回调函数，记录每代信息"""
            try:
                # 调用原始回调
                if original_callback:
                    try:
                        original_callback(gp, generation)
                    except Exception as e:
                        print_with_pid(f"\n【警告】原始回调执行出错: {str(e)}")
                
                # 记录最佳程序
                if hasattr(gp, '_best_program'):
                    try:
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
                        print_with_pid(f"\n【第{generation}代总结】:")
                        print_with_pid(f"  最佳预测能力(适应度): {best_fitness:.6f}")
                        print_with_pid(f"  最佳数学表达式: {best_expr}")
                        
                        # 添加程序复杂度分析
                        expr_length = best_program.length_
                        print_with_pid(f"  表达式复杂度: {expr_length} 个节点")
                        
                        # 添加简单的程序解释
                        try:
                            operators_count = {op: best_expr.count(op) for op in self.function_set if op in best_expr}
                            if operators_count:
                                print_with_pid(f"  使用的操作符: {', '.join([f'{op}({cnt}次)' for op, cnt in operators_count.items() if cnt > 0])}")
                        except Exception as e:
                            print_with_pid(f"  无法解析操作符使用情况: {str(e)}")
                        
                        print_with_pid(f"  计算耗时: {elapsed:.2f}秒")
                        
                        # 如果是第0代，检查stopping_criteria设置
                        if generation == 0:
                            print_with_pid(f"\n【初始种群分析】")
                            stopping_criteria_value = getattr(gp, 'stopping_criteria', '未找到')
                            print_with_pid(f"  停止条件阈值设置: {stopping_criteria_value}")
                            
                            if hasattr(gp, 'stopping_criteria') and isinstance(gp.stopping_criteria, (int, float)):
                                comparison = '超过' if best_fitness > gp.stopping_criteria else '未超过'
                                print_with_pid(f"  最佳适应度 {best_fitness} 与停止条件比较: {comparison} 停止条件")
                                if best_fitness > gp.stopping_criteria and gp.stopping_criteria > 0:
                                    print_with_pid(f"  【预警】可能过早触发停止条件! 初始适应度({best_fitness})已经超过停止阈值({gp.stopping_criteria})")
                                    print_with_pid(f"  建议: 提高stopping_criteria值或设为0以禁用此条件")
                        
                        # 检查是否只有第0代
                        if generation == 0 and hasattr(gp, '_iterations') and gp._iterations <= 1:
                            print_with_pid("\n【遗传算法异常】只执行了第0代就停止了!")
                            print_with_pid("  可能原因:")
                            print_with_pid("  1. stopping_criteria设置过低，初始种群就满足了停止条件")
                            print_with_pid("  2. 初始种群包含了特别优秀的个体")
                            print_with_pid("  3. 算法内部逻辑问题导致提前终止")
                            print_with_pid("  建议操作:")
                            print_with_pid("  1. 设置stopping_criteria=0禁用早停")
                            print_with_pid("  2. 增大random_state值或设为None使用随机种子")
                            print_with_pid("  3. 调大p_crossover和p_mutation增加种群多样性")
                            
                        # 当演化继续到第10代时，确认早停已禁用
                        if generation == 10:
                            print_with_pid("\n【正常进化确认】")
                            print_with_pid("  演化已成功进行了10代，早停机制未触发!")
                            
                        # 每10代输出一次进度
                        if generation > 0 and generation % 10 == 0:
                            print_with_pid(f"\n===== 演化进度: {generation}/{gp.generations}代 ({(generation/gp.generations*100):.1f}%) =====")
                            if len(self.evolution_history) >= 2:
                                try:
                                    first_gen = self.evolution_history[0]
                                    current_gen = self.evolution_history[-1]
                                    if first_gen['best_expr'] == current_gen['best_expr']:
                                        print_with_pid("【演化停滞】当前最佳表达式与第0代相同，可能陷入局部最优")
                                        print_with_pid("  建议: 调整突变率或使用不同随机种子")
                                    else:
                                        print_with_pid("【演化正常】最佳表达式已经改变，算法在有效搜索")
                                    
                                    improvement = current_gen['best_fitness'] - first_gen['best_fitness']
                                    print_with_pid(f"  与第0代相比，适应度提升: {improvement:.6f}")
                                except Exception as e:
                                    print_with_pid(f"  无法计算演化进度详情: {str(e)}")
                                
                                try:
                                    # 计算剩余时间
                                    elapsed_per_gen = elapsed / (generation + 1)
                                    remaining_gens = gp.generations - generation
                                    estimated_remaining = elapsed_per_gen * remaining_gens
                                    
                                    # 格式化为小时:分钟:秒
                                    hours, remainder = divmod(estimated_remaining, 3600)
                                    minutes, seconds = divmod(remainder, 60)
                                    time_str = ""
                                    if hours > 0:
                                        time_str += f"{int(hours)}小时"
                                    if minutes > 0 or hours > 0:
                                        time_str += f"{int(minutes)}分钟"
                                    time_str += f"{int(seconds)}秒"
                                    
                                    print_with_pid(f"  估计剩余时间: {time_str}")
                                except Exception as e:
                                    print_with_pid(f"  无法计算剩余时间: {str(e)}")
                    except Exception as e:
                        print_with_pid(f"\n【警告】处理最佳程序时出错: {str(e)}")
            except Exception as e:
                print_with_pid(f"\n【错误】增强回调执行失败: {str(e)}")
                # 即使发生错误，也要确保程序继续运行
        
        # 替换回调函数
        self.regressor._callback = lambda gp, generation: enhanced_callback(gp, generation)
        
        # 保存原始的fit方法
        original_fit = self.regressor.fit
        
        # 定义增强的fit方法
        def monitored_fit(X, y, *args, **kwargs):
            print_with_pid("\n【调试】开始执行fit方法...")
            
            # 检查stopping_criteria属性
            print_with_pid(f"【调试】stopping_criteria设置为: {getattr(self.regressor, 'stopping_criteria', '未设置')}")
            
            # 拦截_run方法以检测早停
            original_run = getattr(self.regressor, '_run', None)
            
            if original_run:
                def monitored_run(X, y, *args, **kwargs):
                    print_with_pid("\n【调试】开始执行_run方法...")
                    result = original_run(X, y, *args, **kwargs)
                    print_with_pid(f"\n【调试】_run方法完成，返回结果类型: {type(result)}")
                    
                    # 检查迭代次数
                    if hasattr(self.regressor, '_iterations'):
                        print_with_pid(f"【调试】迭代结束后的_iterations值: {self.regressor._iterations}")
                    
                    # 检查最佳程序
                    if hasattr(self.regressor, '_best_program'):
                        print_with_pid(f"【调试】最佳程序表达式: {str(self.regressor._best_program)}")
                        print_with_pid(f"【调试】最佳程序适应度: {self.regressor._best_program.raw_fitness_}")
                    
                    return result
                
                # 替换_run方法
                self.regressor._run = monitored_run
            
            # 执行原始fit方法
            result = original_fit(X, y, *args, **kwargs)
            
            print_with_pid(f"\n【调试】fit方法执行完成")
            if hasattr(self.regressor, '_iterations'):
                print_with_pid(f"  实际执行迭代次数: {self.regressor._iterations}")
                
                # 添加这一行来判断是否触发了早停
                if self.regressor._iterations <= 1:
                    print_with_pid("\n【警告】遗传算法提前终止!")
                    print_with_pid("  只执行了一代就停止，正在检查原因...")
                    
                    # 检查可能的早停原因
                    if hasattr(self.regressor, '_best_program'):
                        fitness = abs(self.regressor._best_program.raw_fitness_)
                        if fitness >= self.stopping_criteria:
                            print_with_pid(f"  【确认】触发了适应度早停条件: 最佳适应度({fitness}) >= 停止阈值({self.stopping_criteria})")
                        else:
                            print_with_pid(f"  【排除】适应度早停条件未触发: 最佳适应度({fitness}) < 停止阈值({self.stopping_criteria})")
                    
                    # 检查固定种子问题
                    print_with_pid(f"  【排查】随机种子设置为: {getattr(self.regressor, 'random_state', '未知')}")
                    print_with_pid(f"  【建议】尝试修改random_state=None以使用不同随机种子")
            
            # 恢复原始_run方法
            if original_run:
                self.regressor._run = original_run
                
            print_with_pid(f"  总耗时: {time.time() - start_time:.2f}秒")
            
            return result
        
        # 替换fit方法
        self.regressor.fit = lambda X, y, *args, **kwargs: monitored_fit(X, y, *args, **kwargs)
        
        # 训练模型
        try:
            self.regressor.fit(X, y)
            
            # 训练后检查
            print_with_pid("\n============= 训练完成信息 =============")
            if hasattr(self.regressor, '_iterations'):
                print_with_pid(f"实际执行代数: {self.regressor._iterations}/{self.regressor.generations}")
                
                if self.regressor._iterations < self.regressor.generations:
                    print_with_pid(f"\n【训练提前结束】实际执行了{self.regressor._iterations}代，少于设定的{self.regressor.generations}代")
                    print_with_pid("可能原因:")
                    print_with_pid("1. 找到满足stopping_criteria的解")
                    print_with_pid("2. 初始种群已包含最优解")
                    print_with_pid("3. 算法内部逻辑提前终止")
            
            # 显示演化历史统计
            if self.evolution_history:
                first_gen = self.evolution_history[0]
                last_gen = self.evolution_history[-1]
                
                print_with_pid("\n【演化历史总结】")
                print_with_pid(f"初始(第0代):")
                print_with_pid(f"  最佳适应度: {first_gen['best_fitness']:.6f}")
                print_with_pid(f"  最佳表达式: {first_gen['best_expr']}")
                
                print_with_pid(f"\n最终(第{len(self.evolution_history)-1}代):")
                print_with_pid(f"  最佳适应度: {last_gen['best_fitness']:.6f}")
                print_with_pid(f"  最佳表达式: {last_gen['best_expr']}")
                
                # 计算改进幅度
                fitness_improvement = last_gen['best_fitness'] - first_gen['best_fitness']
                improvement_percent = (fitness_improvement / first_gen['best_fitness']) * 100 if first_gen['best_fitness'] > 0 else 0
                
                print_with_pid(f"\n【优化效果】")
                print_with_pid(f"  适应度绝对提升: {fitness_improvement:.6f}")
                if fitness_improvement > 0:
                    print_with_pid(f"  适应度相对提升: {improvement_percent:.2f}%")
                
                if first_gen['best_expr'] == last_gen['best_expr'] and len(self.evolution_history) > 1:
                    print_with_pid("\n【重要发现】第一代和最后一代的最佳表达式完全相同!")
                    print_with_pid("这意味着:")
                    print_with_pid("1. 算法在第一代就找到了最优解，后续未能改进")
                    print_with_pid("2. 可能需要调整算法参数以增加种群多样性")
                    print_with_pid("3. 建议尝试不同的随机种子或增大种群规模")
                
                # 总耗时统计
                total_time = last_gen['elapsed_time']
                hours, remainder = divmod(total_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = ""
                if hours > 0:
                    time_str += f"{int(hours)}小时"
                if minutes > 0 or hours > 0:
                    time_str += f"{int(minutes)}分钟"
                time_str += f"{int(seconds)}秒"
                
                print_with_pid(f"\n【执行统计】")
                print_with_pid(f"  总耗时: {time_str}")
                print_with_pid(f"  每代平均耗时: {total_time/len(self.evolution_history):.2f}秒")
                print_with_pid("=========================================")
        
        except Exception as e:
            logging.error(f"遗传算法训练出错: {str(e)}")
            raise
        finally:
            # 恢复原始方法
            self.regressor.fit = original_fit
            self.regressor._callback = original_callback
        
        # 获取最优程序
        best_programs = []
        min_complexity = 2  # 设置最小复杂度，避免过于简单的表达式
        
        # 先按适应度排序，获取前n_best*3个候选
        candidates = sorted(self.regressor._programs[-1], key=lambda p: abs(p.raw_fitness_), reverse=True)[:n_best*3]
        
        # 从候选中筛选有效的程序
        valid_count = 0
        for program in candidates:
            # 检查程序是否有效
            try:
                factor_values = program.execute(X)
                
                # 检查因子值是否有效
                invalid_values = np.isnan(factor_values) | np.isinf(factor_values)
                valid_ratio = 1.0 - np.mean(invalid_values)
                
                # 检查唯一值的比例
                if len(factor_values) > 0:
                    unique_ratio = len(np.unique(factor_values)) / len(factor_values)
                else:
                    unique_ratio = 0
                
                # 如果无效值过多或唯一值过少，跳过
                if valid_ratio < 0.9 or unique_ratio < 0.01 or program.length_ < min_complexity:
                    continue
                
                # 添加到有效程序列表
                factor = {
                    'expression': str(program),
                    'fitness': abs(program.raw_fitness_),
                    'complexity': program.length_,
                    'program': program
                }
                best_programs.append(factor)
                valid_count += 1
                
                # 如果已经找到足够的有效程序，停止循环
                if valid_count >= n_best:
                    break
            except Exception as e:
                logging.warning(f"程序执行错误: {str(e)}, 表达式: {str(program)}")
                continue
        
        # 如果没有找到有效程序，使用最简单的几个程序
        if not best_programs:
            logging.warning("未找到有效因子，使用最简单的程序")
            for program in sorted(self.regressor._programs[-1], key=lambda p: p.length_)[:n_best]:
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
            
        # 再次按评估指标排序，使用综合得分
        for factor in best_programs:
            # 计算综合得分: |IC| * 稳定性 * (做多收益 + 做空收益) / 复杂度
            ic_abs = abs(factor['ic'])
            stability = abs(factor['stability'])  # 使用绝对值避免负稳定性
            total_returns = factor['long_returns'] + factor['short_returns']
            complexity_penalty = 1 / max(1, factor['complexity'] / 5)  # 复杂度惩罚
            
            # 避免NaN和无穷大
            if np.isnan(ic_abs) or np.isinf(ic_abs): ic_abs = 0.0
            if np.isnan(stability) or np.isinf(stability): stability = 0.0
            if np.isnan(total_returns) or np.isinf(total_returns): total_returns = 0.0
            
            # 避免总收益为负的情况
            if total_returns <= 0:
                total_returns = 0.0001  # 给一个很小的正值
                
            factor['score'] = ic_abs * stability * total_returns * complexity_penalty * 1000
            
            # 如果得分无效，设为0
            if np.isnan(factor['score']) or np.isinf(factor['score']):
                factor['score'] = 0.0
                
        # 按照评分排序
        best_programs.sort(key=lambda x: x['score'], reverse=True)
        
        return best_programs
    
    def evaluate_factor(self, factor: Dict, 
                       data: pd.DataFrame,
                       forward_period: int = 24) -> Dict[str, float]:
        """评估因子表现"""
        X = self._prepare_features(data)
        factor_values = factor['program'].execute(X)
        
        # 处理无效值
        factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算未来收益
        future_returns = data['close'].shift(-forward_period) / data['close'] - 1
        
        # 确保数据长度匹配
        valid_idx = ~np.isnan(future_returns)
        factor_values = factor_values[:-forward_period][valid_idx[:-forward_period]]
        future_returns = future_returns[:-forward_period][valid_idx[:-forward_period]]
        
        # 移除可能的无限值
        valid_data = ~np.isnan(factor_values) & ~np.isinf(factor_values) & ~np.isnan(future_returns) & ~np.isinf(future_returns)
        factor_values = factor_values[valid_data]
        future_returns = future_returns[valid_data]
        
        # 如果数据量不足，返回默认值
        if len(factor_values) < 100 or len(np.unique(factor_values)) < 10:
            return {
                'ic': 0.0,
                'stability': 0.0,
                'complexity': factor['complexity'],
                'long_returns': 0.0,
                'short_returns': 0.0,
                'valid_data_ratio': 0.0 if len(valid_idx) == 0 else sum(valid_data) / len(valid_idx)
            }
            
        # 计算IC值
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(factor_values, future_returns)
            ic = corr_matrix[0,1] if not np.isnan(corr_matrix[0,1]) else 0.0
        
        # 确保IC不是NaN
        if np.isnan(ic) or np.isinf(ic):
            ic = 0.0
        
        # 计算因子稳定性（使用原始因子值计算稳定性）
        if len(factor_values) > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                stability_corr = np.corrcoef(factor_values[1:], factor_values[:-1])
                stability = stability_corr[0,1] if not np.isnan(stability_corr[0,1]) else 0.0
                if np.isnan(stability) or np.isinf(stability):
                    stability = 0.0
        else:
            stability = 0.0
        
        # 计算因子收益率
        if len(factor_values) >= 10:  # 确保有足够样本计算分位数
            long_threshold = np.percentile(factor_values, 80)
            short_threshold = np.percentile(factor_values, 20)
            
            long_positions = factor_values > long_threshold
            short_positions = factor_values < short_threshold
            
            if np.any(long_positions):
                long_returns = future_returns[long_positions].mean()
                if np.isnan(long_returns) or np.isinf(long_returns):
                    long_returns = 0.0
            else:
                long_returns = 0.0
                
            if np.any(short_positions):
                short_returns = -future_returns[short_positions].mean()
                if np.isnan(short_returns) or np.isinf(short_returns):
                    short_returns = 0.0
            else:
                short_returns = 0.0
        else:
            long_returns = 0.0
            short_returns = 0.0
        
        return {
            'ic': ic,
            'stability': stability,
            'complexity': factor['complexity'],
            'long_returns': long_returns,
            'short_returns': short_returns,
            'valid_data_ratio': sum(valid_data) / len(valid_idx) if len(valid_idx) > 0 else 0.0
        }

    def _explain_metrics(self):
        """解释进化过程中显示的各项指标"""
        print_with_pid("\n=================== 遗传算法指标说明 ===================")
        print_with_pid("代数：当前进化的代数（从0开始计数）")
        print_with_pid("  - 第0代是初始随机生成的种群")
        print_with_pid("  - 数字越大表示进化越多轮次")
        print_with_pid("")
        print_with_pid("平均长度：当前种群中所有程序的平均复杂度")
        print_with_pid("  - 值越大意味着表达式包含更多的运算符和变量")
        print_with_pid("  - 一般会随着进化逐渐下降，算法倾向于寻找简单的解")
        print_with_pid("")
        print_with_pid("平均适应度：当前种群所有程序的平均表现分数")
        print_with_pid("  - 值越大表示程序平均预测能力越强")
        print_with_pid("  - 一般会随着进化逐渐下降（因为我们在最小化误差）")
        print_with_pid("")
        print_with_pid("最佳长度：当前最佳程序的复杂度")
        print_with_pid("  - 表示当前找到的最好程序包含的节点数量")
        print_with_pid("")
        print_with_pid("最佳适应度：当前最佳程序的表现分数")
        print_with_pid("  - 值越大表示预测能力越强")
        print_with_pid("  - 不变表示未找到更好的程序")
        print_with_pid("")
        print_with_pid("OOB适应度：对未训练数据的评估分数（通常不启用）")
        print_with_pid("")
        print_with_pid("预计剩余时间：完成所有进化预计还需要的时间")
        print_with_pid("==================================================\n")