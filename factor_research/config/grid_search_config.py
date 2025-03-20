"""
网格搜索配置模块
"""

# 参数网格定义
PARAM_GRID = {
    # 未来收益预测周期，单位为K线数量
    'forward_period': [12, 24, 72, 120, 288],  # 1小时, 2小时, 6小时, 10小时, 24小时
    
    # 遗传算法参数
    'generations': [50, 100, 200],  # 进化代数
    'population_size': [1000, 3000, 5000],  # 种群大小
    'tournament_size': [10, 20, 30]  # 锦标赛大小
}

# 特殊参数组合，直接添加到参数列表中
SPECIAL_COMBINATIONS = [
    # 特殊组合1：小种群快速迭代
    {
        'forward_period': 72, 
        'generations': 30, 
        'population_size': 500, 
        'tournament_size': 5
    },
    # 特殊组合2：大种群深度探索
    {
        'forward_period': 72, 
        'generations': 300, 
        'population_size': 8000, 
        'tournament_size': 20
    }
]

# 固定参数，对所有网格点使用相同的值
FIXED_PARAMS = {
    # 早停设置
    'stopping_criteria': 0.001,  # 适应度改善阈值
    'early_stopping': 30,  # 连续代数无改善时停止
    
    # 随机种子
    # 'random_state': 42,  # 固定随机种子以保证可重复性
    # 改为使用当前时间作为种子，增加随机性
    
    # 变异参数 - 总和必须<=1.0
    'p_crossover': 0.4,  # 交叉概率，从0.65降至0.4
    'p_subtree_mutation': 0.2,  # 子树变异概率，从0.25降至0.2
    'p_hoist_mutation': 0.2,  # 提升变异概率，从0.15升至0.2
    'p_point_mutation': 0.2,  # 点变异概率，从0.15升至0.2
    # 当前总和: 0.4 + 0.2 + 0.2 + 0.2 = 1.0，符合要求
    
    # 其他参数
    'const_range': (-3.0, 3.0),  # 常数范围
    'parsimony_coefficient': 0.005,  # 复杂度惩罚系数
    'init_depth': (2, 6),  # 初始树深度范围
} 