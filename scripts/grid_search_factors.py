import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import time
import json
from datetime import datetime
from tqdm import tqdm
import os

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_research.symbolic_miner import SymbolicFactorMiner

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactorGridSearch:
    """
    因子参数网格搜索类
    
    自动测试不同参数组合以寻找最佳因子表现
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "results" / "grid_search"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存所有参数组合的结果
        self.all_results = []
        
        # 参数网格定义
        self.param_grid = {
            "forward_period": [12, 24, 36, 48, 60, 72, 84, 96, 120],
            "generations": [100, 200, 300, 400, 500],
            "population_size": [3000, 5000],
            "tournament_size": [20, 25, 30]
        }
        
    def load_data(self):
        """加载全部5分钟K线数据"""
        logger.info("开始加载数据文件...")
        
        # 获取所有5分钟K线数据文件
        data_files = list(self.project_root.glob("data/kline/ETHUSDT_5m_*.csv"))
        
        if not data_files:
            logger.error("未找到5分钟K线数据文件")
            logger.error("请先运行 scripts/download_history.py 下载数据")
            return None
            
        logger.info(f"找到 {len(data_files)} 个数据文件")
        
        # 合并所有文件数据
        all_data = []
        for file_path in tqdm(data_files, desc="加载数据文件"):
            df = pd.read_csv(file_path)
            all_data.append(df)
        
        # 合并并排序去重
        data = pd.concat(all_data)
        data = data.sort_values('timestamp').drop_duplicates()
        
        logger.info(f"数据加载完成，共 {len(data)} 条记录")
        logger.info(f"数据时间范围: {min(data['timestamp'])} 到 {max(data['timestamp'])}")
        
        return data
    
    def evaluate_factor(self, factor):
        """评估因子质量的综合得分"""
        # 计算综合分数: |IC| * 稳定性 * (做多收益 + 做空收益) / 复杂度
        ic_abs = abs(factor['ic'])
        stability = abs(factor['stability'])  # 使用绝对值避免负稳定性
        total_returns = factor['long_returns'] + factor['short_returns']
        complexity_penalty = 1 / max(1, factor['complexity'] / 2)  # 复杂度惩罚
        
        # 避免总收益为负的情况
        if total_returns <= 0:
            total_returns = 0.0001  # 给一个很小的正值
            
        score = ic_abs * stability * total_returns * complexity_penalty * 1000
        return score
    
    def run_grid_search(self):
        """执行网格搜索"""
        # 加载数据
        data = self.load_data()
        if data is None:
            return False
            
        # 创建结果文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"grid_search_results_{timestamp}.json"
        
        # 计算总组合数
        total_combinations = (
            len(self.param_grid["forward_period"]) * 
            len(self.param_grid["generations"]) *
            len(self.param_grid["population_size"]) *
            len(self.param_grid["tournament_size"])
        )
        
        logger.info(f"开始网格搜索，共 {total_combinations} 种参数组合")
        
        # 用于跟踪进度的计数器
        current_combination = 0
        
        # 开始网格搜索
        for forward_period in self.param_grid["forward_period"]:
            for generations in self.param_grid["generations"]:
                for population_size in self.param_grid["population_size"]:
                    for tournament_size in self.param_grid["tournament_size"]:
                        current_combination += 1
                        
                        # 显示进度
                        logger.info(f"参数组合 {current_combination}/{total_combinations}: "
                                   f"forward_period={forward_period}, "
                                   f"generations={generations}, "
                                   f"population_size={population_size}, "
                                   f"tournament_size={tournament_size}")
                        
                        # 创建挖掘器
                        miner = SymbolicFactorMiner(
                            population_size=population_size,
                            generations=generations,
                            tournament_size=tournament_size,
                            stopping_criteria=0.001,
                            early_stopping=30,
                            const_range=(-2.0, 2.0)
                        )
                        
                        try:
                            # 挖掘因子
                            start_time = time.time()
                            factors = miner.mine_factors(
                                data=data,
                                n_best=5,  # 获取5个最佳因子
                                forward_period=forward_period
                            )
                            end_time = time.time()
                            
                            # 计算每个因子的综合得分
                            for factor in factors:
                                factor['score'] = self.evaluate_factor(factor)
                            
                            # 按得分排序
                            factors.sort(key=lambda x: x['score'], reverse=True)
                            
                            # 取最佳因子
                            best_factor = factors[0]
                            
                            # 记录结果
                            result = {
                                "forward_period": forward_period,
                                "generations": generations,
                                "population_size": population_size,
                                "tournament_size": tournament_size,
                                "best_factor": {
                                    "expression": best_factor['expression'],
                                    "ic": best_factor['ic'],
                                    "stability": best_factor['stability'],
                                    "long_returns": best_factor['long_returns'],
                                    "short_returns": best_factor['short_returns'],
                                    "complexity": best_factor['complexity'],
                                    "score": best_factor['score']
                                },
                                "all_factors": [{
                                    "expression": f['expression'],
                                    "ic": f['ic'],
                                    "stability": f['stability'],
                                    "long_returns": f['long_returns'],
                                    "short_returns": f['short_returns'],
                                    "complexity": f['complexity'],
                                    "score": f['score']
                                } for f in factors],
                                "execution_time": end_time - start_time
                            }
                            
                            self.all_results.append(result)
                            
                            # 定期保存结果
                            with open(results_file, 'w', encoding='utf-8') as f:
                                json.dump(self.all_results, f, indent=2, ensure_ascii=False)
                                
                            logger.info(f"最佳因子 - 表达式: {best_factor['expression']}, "
                                       f"得分: {best_factor['score']:.4f}, "
                                       f"IC: {best_factor['ic']:.4f}, "
                                       f"稳定性: {best_factor['stability']:.4f}")
                                       
                        except Exception as e:
                            logger.error(f"参数组合运行错误: {str(e)}")
                            
        # 按得分排序所有结果
        self.all_results.sort(key=lambda x: x['best_factor']['score'], reverse=True)
        
        # 保存最终结果
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
            
        # 生成报告
        self.generate_report(timestamp)
        
        return True
    
    def generate_report(self, timestamp):
        """生成网格搜索报告"""
        report_file = self.results_dir / f"grid_search_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 因子参数网格搜索报告\n\n")
            f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 参数网格\n\n")
            for param, values in self.param_grid.items():
                f.write(f"- **{param}**: {values}\n")
            f.write("\n")
            
            f.write("## 前10名最佳参数组合\n\n")
            f.write("| 排名 | forward_period | generations | population_size | tournament_size | 因子表达式 | 得分 | IC | 稳定性 | 做多收益 | 做空收益 | 复杂度 |\n")
            f.write("|------|---------------|-------------|-----------------|-----------------|------------|------|----|--------|----------|----------|--------|\n")
            
            for i, result in enumerate(self.all_results[:10], 1):
                factor = result['best_factor']
                f.write(f"| {i} | {result['forward_period']} | {result['generations']} | ")
                f.write(f"{result['population_size']} | {result['tournament_size']} | ")
                f.write(f"`{factor['expression']}` | {factor['score']:.4f} | ")
                f.write(f"{factor['ic']:.4f} | {factor['stability']:.4f} | ")
                f.write(f"{factor['long_returns']:.4f} | {factor['short_returns']:.4f} | ")
                f.write(f"{factor['complexity']} |\n")
            
            f.write("\n## 详细结果分析\n\n")
            
            # 按forward_period分组分析
            f.write("### 按预测周期(forward_period)分组\n\n")
            period_groups = {}
            for result in self.all_results:
                period = result['forward_period']
                if period not in period_groups:
                    period_groups[period] = []
                period_groups[period].append(result['best_factor']['score'])
            
            f.write("| forward_period | 平均得分 | 最高得分 | 结果数量 |\n")
            f.write("|---------------|----------|----------|----------|\n")
            
            for period, scores in sorted(period_groups.items()):
                f.write(f"| {period} | {np.mean(scores):.4f} | {max(scores):.4f} | {len(scores)} |\n")
            
            # 按generations分组分析
            f.write("\n### 按进化代数(generations)分组\n\n")
            gen_groups = {}
            for result in self.all_results:
                gen = result['generations']
                if gen not in gen_groups:
                    gen_groups[gen] = []
                gen_groups[gen].append(result['best_factor']['score'])
            
            f.write("| generations | 平均得分 | 最高得分 | 结果数量 |\n")
            f.write("|-------------|----------|----------|----------|\n")
            
            for gen, scores in sorted(gen_groups.items()):
                f.write(f"| {gen} | {np.mean(scores):.4f} | {max(scores):.4f} | {len(scores)} |\n")
                
            # 总结
            f.write("\n## 结论和建议\n\n")
            
            # 找出最佳forward_period
            best_period = max(period_groups.items(), key=lambda x: np.mean(x[1]))[0]
            f.write(f"- 最佳预测周期(forward_period): **{best_period}**\n")
            
            # 找出最佳generations
            best_gen = max(gen_groups.items(), key=lambda x: np.mean(x[1]))[0]
            f.write(f"- 最佳进化代数(generations): **{best_gen}**\n")
            
            # 整体最佳参数组合
            best_result = self.all_results[0]
            f.write("\n最佳因子来自参数组合:\n")
            f.write(f"- forward_period = {best_result['forward_period']}\n")
            f.write(f"- generations = {best_result['generations']}\n")
            f.write(f"- population_size = {best_result['population_size']}\n")
            f.write(f"- tournament_size = {best_result['tournament_size']}\n")
            
            # 最佳因子信息
            best_factor = best_result['best_factor']
            f.write("\n最佳因子:\n")
            f.write(f"- 表达式: `{best_factor['expression']}`\n")
            f.write(f"- 得分: {best_factor['score']:.4f}\n")
            f.write(f"- IC: {best_factor['ic']:.4f}\n")
            f.write(f"- 稳定性: {best_factor['stability']:.4f}\n")
            f.write(f"- 做多收益: {best_factor['long_returns']:.4f}\n")
            f.write(f"- 做空收益: {best_factor['short_returns']:.4f}\n")
            f.write(f"- 复杂度: {best_factor['complexity']}\n")
        
        logger.info(f"报告已生成: {report_file}")

if __name__ == "__main__":
    searcher = FactorGridSearch()
    searcher.run_grid_search() 