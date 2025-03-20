"""
网格搜索执行脚本
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import time
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import pickle

# 导入项目模块
from factor_research.symbolic_miner import SymbolicFactorMiner
from factor_research.data_loader import load_data_files
from factor_research.config.grid_search_config import PARAM_GRID, SPECIAL_COMBINATIONS, FIXED_PARAMS

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PID:%(process)d] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建进程ID日志过滤器
class PidLoggingFilter(logging.Filter):
    def filter(self, record):
        record.pid = os.getpid()
        return True

# 获取logger并添加过滤器
logger.addFilter(PidLoggingFilter())

# 创建带进程ID的打印函数
def print_with_pid(*args, **kwargs):
    """带进程ID的打印函数"""
    pid = os.getpid()
    prefix = f"[PID-{pid}] "
    
    # 构造新的参数
    if args and isinstance(args[0], str):
        new_args = (prefix + args[0],) + args[1:]
        print(*new_args, **kwargs, flush=True)  # 添加flush=True确保立即输出
    else:
        print(prefix, end="", flush=True)
        print(*args, **kwargs, flush=True)

# 重定向系统标准输出和标准错误
original_print = print
sys.stdout.write_original = sys.stdout.write

def custom_write(text):
    """自定义写入函数，添加进程ID"""
    pid = os.getpid()
    if text.strip() and not text.startswith('[PID-'):
        return sys.stdout.write_original(f"[PID-{pid}] {text}")
    return sys.stdout.write_original(text)

# 修改部分共用的日志处理
class PidLoggingFilter(logging.Filter):
    """添加进程ID到日志记录"""
    def filter(self, record):
        record.pid = os.getpid()
        return True

logger.addFilter(PidLoggingFilter())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PID-%(pid)s] - %(levelname)s - %(message)s'
)

def generate_param_combinations():
    """生成所有参数组合"""
    param_combinations = []
    for forward_period in PARAM_GRID['forward_period']:
        for generations in PARAM_GRID['generations']:
            for population_size in PARAM_GRID['population_size']:
                for tournament_size in PARAM_GRID['tournament_size']:
                    param_combinations.append({
                        'forward_period': forward_period,
                        'generations': generations,
                        'population_size': population_size,
                        'tournament_size': tournament_size
                    })
    
    # 添加特殊组合
    param_combinations.extend(SPECIAL_COMBINATIONS)
    return param_combinations

def process_mining_result(result):
    """处理挖掘结果"""
    if isinstance(result, list) and len(result) > 0:
        # 如果结果是列表（多个因子），取第一个因子
        best_factor = result[0]
        return {
            'best_fitness': best_factor.get('fitness', float('inf')),
            'best_program': str(best_factor.get('program', '')),
            'execution_time': best_factor.get('execution_time', 0),
            'generations_completed': best_factor.get('generations_completed', 0),
            'status': 'success'
        }
    elif isinstance(result, dict):
        # 如果结果是字典（单个因子）
        return {
            'best_fitness': result.get('fitness', float('inf')),
            'best_program': str(result.get('program', '')),
            'execution_time': result.get('execution_time', 0),
            'generations_completed': result.get('generations_completed', 0),
            'status': 'success'
        }
    else:
        raise ValueError(f"无法处理的结果格式: {type(result)}")

def save_results(results):
    """保存网格搜索结果"""
    # 创建结果目录
    results_dir = Path("results/grid_search")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        results_dir / f"grid_search_results_{timestamp}.csv",
        index=False
    )
    
    # 生成报告
    report_path = results_dir / f"grid_search_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 网格搜索结果报告\n\n")
        f.write(f"## 执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 写入参数配置
        f.write("## 参数配置\n\n")
        f.write("### 参数网格\n```python\n")
        f.write(str(PARAM_GRID))
        f.write("\n```\n\n")
        
        # 写入最佳结果
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['best_fitness'])
            f.write("## 最佳结果\n\n")
            f.write(f"- Forward Period: {best_result['params']['forward_period']}\n")
            f.write(f"- Generations: {best_result['params']['generations']}\n")
            f.write(f"- Population Size: {best_result['params']['population_size']}\n")
            f.write(f"- Tournament Size: {best_result['params']['tournament_size']}\n")
            f.write(f"- Best Fitness: {best_result['best_fitness']}\n")
            f.write(f"- Best Program: {best_result['best_program']}\n")
        
        # 写入统计信息
        f.write("\n## 执行统计\n\n")
        f.write(f"- 总参数组合数: {len(results)}\n")
        f.write(f"- 成功执行数: {len(successful_results)}\n")
        f.write(f"- 失败数: {len(results) - len(successful_results)}\n")

def initialize_process():
    """初始化进程环境"""
    # 修改标准输出函数
    sys.stdout.write = custom_write
    
    # 修改SymbolicFactorMiner类的打印行为
    original_init = SymbolicFactorMiner.__init__
    
    def wrapped_init(self, *args, **kwargs):
        # 调用原始初始化
        original_init(self, *args, **kwargs)
        
        # 保存原始的_callback方法
        original_callback = getattr(self.regressor, '_callback', None)
        
        # 替换_callback方法，添加进程ID
        if original_callback:
            def wrapped_callback(gp, generation):
                # 添加进程ID到输出
                pid = os.getpid()
                print(f"\n[PID-{pid}] ===== 进化代数 {generation} =====")
                return original_callback(gp, generation)
            
            self.regressor._callback = wrapped_callback
    
    # 替换初始化方法
    SymbolicFactorMiner.__init__ = wrapped_init

def _single_search(args):
    """在独立进程中执行单个参数组合的搜索"""
    params, df = args
    
    # 初始化进程环境
    initialize_process()
    
    pid = os.getpid()
    print_with_pid(f"开始搜索参数组合: forward_period={params['forward_period']}, " + 
               f"generations={params['generations']}, " + 
               f"population_size={params['population_size']}, " + 
               f"tournament_size={params['tournament_size']}")
    
    try:
        # 创建挖掘器实例
        miner = SymbolicFactorMiner(
            population_size=params['population_size'],
            generations=params['generations'],
            tournament_size=params['tournament_size'],
            **FIXED_PARAMS
        )
        
        # 执行挖掘
        result = miner.mine_factors(
            df,
            forward_period=params['forward_period']
        )
        
        # 处理结果
        processed_result = process_mining_result(result)
        processed_result['params'] = params
        print_with_pid(f"搜索完成，最佳程序: {processed_result['best_program']}")
        return processed_result
        
    except Exception as e:
        print_with_pid(f"参数组合执行失败: {str(e)}")
        return {
            'params': params,
            'status': 'failed',
            'error': str(e)
        }

def run_grid_search():
    """执行网格搜索"""
    # 加载数据
    logger.info("开始加载数据文件...")
    df = load_data_files()
    
    if df is None or df.empty:
        logger.error("数据加载失败")
        return
    
    logger.info(f"数据加载完成，共 {len(df)} 条记录")
    logger.info(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
    
    # 生成参数组合
    param_combinations = generate_param_combinations()
    total_combinations = len(param_combinations)
    logger.info(f"开始网格搜索，共 {total_combinations} 种参数组合")
    
    # 准备多进程参数
    process_args = [(params, df) for params in param_combinations]
    
    # 使用进程池执行搜索
    num_processes = mp.cpu_count()  # 使用所有可用的CPU核心
    logger.info(f"使用 {num_processes} 个进程并行执行")
    
    # 设置多进程启动方法为spawn以确保更好的隔离
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(_single_search, process_args),
            total=total_combinations,
            desc="执行网格搜索"
        ))
    
    # 保存结果
    save_results(results)

class FactorGridSearch:
    """因子网格搜索类"""
    
    def __init__(self, data_dir: str = "data/kline", results_dir: str = "results/grid_search"):
        """初始化搜索器"""
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results = []
        
        # 确保结果目录存在
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # 结果文件路径
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"{results_dir}/grid_search_{self.timestamp}.pkl"
        self.report_file = f"{results_dir}/grid_search_report_{self.timestamp}.md"
        
        print_with_pid(f"初始化网格搜索，结果将保存到: {self.results_file}")
        print_with_pid(f"最终报告将保存到: {self.report_file}")
    
    def _load_data(self) -> pd.DataFrame:
        """加载所有5分钟K线数据"""
        print_with_pid(f"加载数据目录: {self.data_dir}")
        # 调用数据加载器获取所有5分钟K线数据
        data_pattern = f"{self.data_dir}/ETHUSDT_5m_*.csv"
        print_with_pid(f"使用数据模式: {data_pattern}")
        data = load_data_files(data_pattern=data_pattern)
        
        if data.empty:
            print_with_pid("警告: 未加载到数据！请检查数据路径和模式是否正确")
            return data
            
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)
        
        print_with_pid(f"数据加载完成，共 {len(data)} 行，时间范围: {data.index[0]} 到 {data.index[-1]}")
        return data
    
    def _single_search(self, params: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """执行单个参数组合的搜索"""
        # 提取参数
        forward_period = params.get("forward_period", 12)
        population_size = params.get("population_size", 1000)
        generations = params.get("generations", 100)
        tournament_size = params.get("tournament_size", 20)
        p_crossover = params.get("p_crossover", 0.7)
        p_subtree_mutation = params.get("p_subtree_mutation", 0.1)
        p_hoist_mutation = params.get("p_hoist_mutation", 0.05)
        p_point_mutation = params.get("p_point_mutation", 0.05)
        parsimony_coefficient = params.get("parsimony_coefficient", 0.01)
        init_depth = params.get("init_depth", (2, 6))
        stopping_criteria = params.get("stopping_criteria", 0.001)  # 设置stopping_criteria为0.001
        n_best = params.get("n_best", 5)
        
        # 打印当前参数组合
        print_with_pid("\n" + "="*50)
        print_with_pid(f"开始搜索参数组合:")
        print_with_pid(f"  预测周期: {forward_period}")
        print_with_pid(f"  种群大小: {population_size}")
        print_with_pid(f"  进化代数: {generations}")
        print_with_pid(f"  锦标赛大小: {tournament_size}")
        print_with_pid(f"  交叉概率: {p_crossover}")
        print_with_pid(f"  子树变异概率: {p_subtree_mutation}")
        print_with_pid(f"  提升变异概率: {p_hoist_mutation}")
        print_with_pid(f"  点变异概率: {p_point_mutation}")
        print_with_pid(f"  复杂度惩罚系数: {parsimony_coefficient}")
        print_with_pid(f"  初始深度范围: {init_depth}")
        print_with_pid(f"  停止条件: {stopping_criteria}")
        print_with_pid("="*50)
        
        # 创建因子挖掘器实例
        start_time = time.time()
        miner = SymbolicFactorMiner(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            parsimony_coefficient=parsimony_coefficient,
            init_depth=init_depth,
            random_state=None  # 使用None确保每次运行都有不同的随机种子
        )
        
        # 执行因子挖掘
        try:
            factors = miner.mine_factors(data, n_best=n_best, forward_period=forward_period)
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            # 构造结果
            result = {
                "params": params,
                "factors": factors,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # 打印结果摘要
            print_with_pid("\n" + "="*50)
            print_with_pid(f"参数组合搜索完成，耗时: {elapsed_time:.2f}秒")
            print_with_pid(f"找到因子数量: {len(factors)}")
            for i, factor in enumerate(factors):
                print_with_pid(f"因子 {i+1}:")
                print_with_pid(f"  表达式: {factor['expression']}")
                print_with_pid(f"  预测能力(IC): {factor['ic']:.4f}")
                print_with_pid(f"  复杂度: {factor['complexity']}")
            print_with_pid("="*50)
            
            return result
            
        except Exception as e:
            print_with_pid(f"搜索过程中发生错误: {str(e)}")
            return {
                "params": params,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    # 添加辅助函数，代替lambda
    def _process_param_search(self, param_data_tuple):
        """处理单个参数组合的搜索，作为可序列化函数替代lambda
        
        Args:
            param_data_tuple: 包含参数和数据的元组 (params, data)
            
        Returns:
            搜索结果字典
        """
        params, data = param_data_tuple
        return self._single_search(params, data)
    
    def run_grid_search(self):
        """执行网格搜索"""
        print_with_pid("开始网格搜索...")
        
        # 加载数据
        data = self._load_data()
        
        if data.empty:
            print_with_pid("错误: 未能加载数据，无法执行网格搜索")
            return
        
        # 创建参数组合列表
        param_combinations = []
        
        # 从配置中生成网格参数组合
        for forward_period in PARAM_GRID.get("forward_period", [12]):
            for generations in PARAM_GRID.get("generations", [100]):
                for population_size in PARAM_GRID.get("population_size", [1000]):
                    for tournament_size in PARAM_GRID.get("tournament_size", [20]):
                        # 合并固定参数和可变参数
                        params = {
                            "forward_period": forward_period,
                            "generations": generations,
                            "population_size": population_size,
                            "tournament_size": tournament_size,
                            **FIXED_PARAMS
                        }
                        param_combinations.append(params)
        
        # 添加特殊组合
        for special_combo in SPECIAL_COMBINATIONS:
            # 确保特殊组合中包含所有必要的固定参数
            combo = {**FIXED_PARAMS, **special_combo}
            param_combinations.append(combo)
        
        print_with_pid(f"生成了 {len(param_combinations)} 个参数组合")
        
        # 准备任务数据元组
        process_args = [(params, data) for params in param_combinations]
        
        # 使用多进程执行搜索
        results = []
        with mp.Pool(processes=min(mp.cpu_count(), len(param_combinations)), initializer=initialize_process) as pool:
            # 使用imap有序地获取结果
            for result in pool.imap(self._process_param_search, process_args):
                results.append(result)
                # 实时保存结果 - 但不生成报告
                with open(self.results_file, 'wb') as f:
                    pickle.dump(results, f)
                
                # 打印结果进度
                print_with_pid(f"完成进度: {len(results)}/{len(param_combinations)} 组合")
        
        # 所有搜索结束后，生成一次最终报告
        print_with_pid("所有参数组合搜索完成，开始生成最终报告...")
        self._generate_report(results)
        
        print_with_pid("网格搜索完成!")
        print_with_pid(f"结果已保存到: {self.results_file}")
        print_with_pid(f"最终报告已生成: {self.report_file}")
    
    def _generate_report(self, results: List[Dict]):
        """生成网格搜索报告"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("# 因子网格搜索报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 总体统计
            successful_runs = [r for r in results if 'error' not in r]
            f.write(f"## 总体统计\n\n")
            f.write(f"- 参数组合总数: {len(results)}\n")
            f.write(f"- 成功完成组合: {len(successful_runs)}\n")
            f.write(f"- 失败组合: {len(results) - len(successful_runs)}\n\n")
            
            if successful_runs:
                # 按IC排序的最佳因子
                all_factors = []
                for result in successful_runs:
                    if 'factors' in result:
                        for factor in result['factors']:
                            factor['params'] = result['params']
                            all_factors.append(factor)
                
                # 按IC排序
                all_factors.sort(key=lambda x: x['ic'], reverse=True)
                
                f.write(f"## 最佳因子 (按IC排序)\n\n")
                for i, factor in enumerate(all_factors[:10]):  # 只显示前10个
                    f.write(f"### 因子 {i+1}\n\n")
                    f.write(f"- 表达式: `{factor['expression']}`\n")
                    f.write(f"- 预测能力(IC): {factor['ic']:.4f}\n")
                    f.write(f"- 稳定性: {factor['stability']:.4f}\n")
                    f.write(f"- 多头收益: {factor.get('long_returns', 0.0):.4f}\n")
                    f.write(f"- 空头收益: {factor.get('short_returns', 0.0):.4f}\n")
                    f.write(f"- 复杂度: {factor['complexity']}\n")
                    f.write(f"- 参数组合:\n")
                    for k, v in factor['params'].items():
                        if k != 'n_best':  # 忽略n_best参数
                            f.write(f"  - {k}: {v}\n")
                    f.write("\n")
                
                # 按参数组合展示结果
                f.write(f"## 各参数组合结果\n\n")
                for i, result in enumerate(results):
                    f.write(f"### 组合 {i+1}\n\n")
                    f.write(f"#### 参数:\n\n")
                    for k, v in result['params'].items():
                        f.write(f"- {k}: {v}\n")
                    f.write(f"\n#### 运行时间: {result.get('elapsed_time', 0):.2f}秒\n\n")
                    
                    if 'error' in result:
                        f.write(f"#### 错误:\n\n")
                        f.write(f"```\n{result['error']}\n```\n\n")
                    elif 'factors' in result:
                        f.write(f"#### 找到的因子:\n\n")
                        for j, factor in enumerate(result['factors']):
                            f.write(f"##### 因子 {j+1}:\n\n")
                            f.write(f"- 表达式: `{factor['expression']}`\n")
                            f.write(f"- 预测能力(IC): {factor['ic']:.4f}\n")
                            f.write(f"- 稳定性: {factor['stability']:.4f}\n")
                            f.write(f"- 多头收益: {factor.get('long_returns', 0.0):.4f}\n")
                            f.write(f"- 空头收益: {factor.get('short_returns', 0.0):.4f}\n")
                            f.write(f"- 复杂度: {factor['complexity']}\n\n")

if __name__ == "__main__":
    # 执行网格搜索
    searcher = FactorGridSearch()
    searcher.run_grid_search() 