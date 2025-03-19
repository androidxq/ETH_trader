import pandas as pd
from pathlib import Path
import logging
import sys
from factor_research.symbolic_miner import SymbolicFactorMiner
from tqdm import tqdm
import time
import numpy as np

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_symbolic_mining():
    """
    测试符号回归因子挖掘功能
    
    符号回归基于遗传算法，能够自动发现数据中隐含的规律并用数学表达式表示。
    该函数主要步骤:
    1. 加载历史K线数据
    2. 初始化因子挖掘器
    3. 使用遗传算法挖掘因子
    4. 评估和展示获得的因子
    """
    # 获取项目根目录 - 使用相对路径以适应不同环境
    project_root = Path(__file__).parent.parent
    
    # 获取所有5分钟K线数据文件
    data_files = list(project_root.glob("data/kline/ETHUSDT_5m_*.csv"))
    
    if not data_files:
        print(f"错误: 未找到5分钟K线数据文件")
        print(f"请先运行 scripts/download_history.py 下载数据")
        return
        
    print(f"找到 {len(data_files)} 个数据文件")
    
    # 合并所有文件数据
    all_data = []
    for file_path in tqdm(data_files, desc="加载数据文件"):
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    # 合并并排序去重
    data = pd.concat(all_data)
    data = data.sort_values('timestamp').drop_duplicates()
    
    print(f"数据加载完成，共 {len(data)} 条记录")
    print(f"数据时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    # 创建符号回归因子挖掘器
    # 参数说明：
    miner = SymbolicFactorMiner(
        # population_size: 种群大小，表示每一代中个体的数量
        # - 更大的种群提供更多的解空间探索能力，但会增加计算成本
        # - 通常设置为1000-10000之间，5000是一个适中的值
        population_size=5000,
        
        # generations: 进化代数，表示遗传算法迭代的轮数
        # - 更多的代数通常会产生更好的结果，但有过拟合风险
        # - 通常设置为100-500之间，200是一个较好的平衡点
        generations=200,
        
        # tournament_size: 锦标赛大小，控制选择压力
        # - 较大的值会增加选择最佳个体的概率，但可能导致早熟收敛
        # - 通常为种群大小的1-5%，20-30是常见值
        tournament_size=25,
        
        # stopping_criteria: 停止条件，当适应度改善低于此值时提前停止
        # - 设置较小的值可以在收敛后节省计算资源
        # - 设置为0.001表示当适应度改善小于0.001时停止
        stopping_criteria=0.001,
        
        # early_stopping: 早停代数，连续多少代没有改善就停止
        # - 防止过拟合，通常设置为20-50
        early_stopping=30,
        
        # const_range: 常数项取值范围
        # - 控制表达式中常数的数值范围
        const_range=(-2.0, 2.0)
        
        # 以下参数不被SymbolicFactorMiner类直接支持，已移除
        # p_crossover=0.7,
        # p_subtree_mutation=0.2,
        # p_hoist_mutation=0.05,
        # p_point_mutation=0.1,
        # metric='spearman',
        # verbose=2
    )
    
    print("开始挖掘因子...")
    
    # 定义一个更复杂的适应度函数
    def custom_fitness(y, y_pred, w):
        # 使用多个指标组合评估
        ic = np.corrcoef(y, y_pred)[0, 1]
        mae = np.mean(np.abs(y - y_pred))
        return (np.abs(ic) * 10) - mae
    
    # 挖掘因子
    # 参数说明:
    factors = miner.mine_factors(
        # data: 输入数据，包含OHLCV等价格信息的DataFrame
        data=data,
        
        # n_best: 返回的最佳因子数量
        # - 通常设置为3-10之间，取决于需要的因子数量
        n_best=5,
        
        # forward_period: 未来收益计算周期
        # - 对于5分钟数据，设置为比1小时K线(24)更长的值，如288代表一天
        # - 该值应根据交易周期和策略持仓时间调整
        forward_period=72  # 约6小时预测窗口 (72 * 5分钟)
        # 以下参数不被mine_factors方法支持，已移除
        # random_state=int(time.time()),
        # fitness_func=custom_fitness
    )
    
    print(f"因子挖掘完成，获得 {len(factors)} 个因子")
    
    # 打印结果
    for i, factor in enumerate(factors, 1):
        print(f"\n因子 {i}:")
        # expression: 符号表达式，表示因子的数学公式
        print(f"表达式: {factor['expression']}")
        
        # ic: 信息系数，衡量因子与未来收益的相关性
        # - 绝对值越大表示预测能力越强，通常|IC|>0.05就有一定价值
        print(f"预测能力(IC): {factor['ic']:.4f}")
        
        # stability: 稳定性，衡量因子值的连续性
        # - 接近1表示稳定性高，通常>0.7为较好
        print(f"稳定性: {factor['stability']:.4f}")
        
        # long_returns: 做多收益，选择因子值高的标的获得的收益率
        print(f"做多收益: {factor['long_returns']:.4f}")
        
        # short_returns: 做空收益，对因子值低的标的做空获得的收益率
        print(f"做空收益: {factor['short_returns']:.4f}")
        
        # complexity: 复杂度，表达式的节点数量
        # - 越小越简单，复杂度高的表达式可能过拟合
        print(f"复杂度: {factor['complexity']}")

if __name__ == "__main__":
    test_symbolic_mining()