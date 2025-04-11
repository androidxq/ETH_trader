def mine_new_factors(data: pd.DataFrame, 
                    factor_type: str = "量价获利因子", 
                    forward_period: int = 24,
                    n_best: int = 3
                   ) -> List[dict]:
    """使用遗传算法挖掘新因子"""
    from factor_research.symbolic_miner import SymbolicFactorMiner
    
    logger.info(f"开始挖掘 {factor_type} 类型的新因子...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 初始化遗传算法引擎
    miner = SymbolicFactorMiner(
        function_set=["add", "sub", "mul", "div", "log", "sqrt", "square", "pow", "exp"],
        windows=[5, 10, 20, 50, 100],
        population_size=100,
        generations=10,
        verbose=True
    )
    
    # 挖掘因子
    factors = miner.mine_factors(
        data=data,
        n_best=n_best,
        forward_period=forward_period,
        transaction_fee=0.1,
        min_trade_return=0.3,
        factor_type=factor_type  # 指定因子类型
    )
    
    logger.info(f"已挖掘 {len(factors)} 个 {factor_type}:")
    for i, factor in enumerate(factors):
        logger.info(f"{i+1}. 表达式: {factor['expression']}")
        logger.info(f"   IC: {factor['ic']:.4f}, Sharpe: {factor['sharpe']:.4f}, Win Rate: {factor['win_rate']:.4f}")
    
    return factors

# 添加一个新的示例函数来展示如何挖掘支撑阻力因子
def mine_support_resistance_factors(data_path: str = "data/eth_1h.csv") -> List[dict]:
    """挖掘支撑阻力因子的示例函数"""
    # 加载数据
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    # 挖掘支撑阻力因子
    support_resistance_factors = mine_new_factors(
        data=df,
        factor_type="支撑阻力因子",
        forward_period=24,  # 24小时预测期
        n_best=3  # 返回前3个最佳因子
    )
    
    return support_resistance_factors 