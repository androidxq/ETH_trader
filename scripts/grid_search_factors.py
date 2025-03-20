"""
网格搜索执行脚本
"""

import logging
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from factor_research.symbolic_miner import SymbolicFactorMiner
from factor_research.data_loader import load_data_files
from factor_research.config.grid_search_config import PARAM_GRID, SPECIAL_COMBINATIONS, FIXED_PARAMS

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    # 存储结果
    results = []
    
    # 遍历所有参数组合
    for i, params in enumerate(param_combinations, 1):
        logger.info(f"参数组合 {i}/{total_combinations}: " + 
                   f"forward_period={params['forward_period']}, " +
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
            results.append(processed_result)
            
        except Exception as e:
            logger.error(f"参数组合 {i} 执行失败: {str(e)}")
            results.append({
                'params': params,
                'status': 'failed',
                'error': str(e)
            })
    
    # 保存结果
    save_results(results)

if __name__ == "__main__":
    run_grid_search() 