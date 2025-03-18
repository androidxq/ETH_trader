import sys
import pandas as pd
from pathlib import Path
import logging

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from factor_research.base_factor import BaseFactor
from factor_research.optimal_points import OptimalPointsFinder
from factor_research.factor_evaluator import FactorEvaluator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleMomentumFactor(BaseFactor):
    """简单动量因子"""
    def __init__(self, window: int = 20):
        super().__init__('momentum')
        self.set_params(window=window)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        window = self.params['window']
        returns = data['close'].pct_change(window)
        self.factor_values = returns
        return returns

def load_test_data():
    """加载测试数据"""
    # 这里使用我们之前下载的数据
    data_path = Path("d:/pythonProject/ETH_trader/data/kline/ETHUSDT_1h_202401.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def test_factor_framework():
    """测试因子研究框架"""
    try:
        # 1. 加载数据
        logging.info("开始加载测试数据...")
        data = load_test_data()
        logging.info(f"成功加载数据，共 {len(data)} 条记录")
        
        # 2. 寻找最优买卖点
        logging.info("开始寻找最优买卖点...")
        finder = OptimalPointsFinder(lookback_window=24, profit_threshold=0.02)
        buy_points, sell_points = finder.find_optimal_points(data.reset_index())
        logging.info(f"找到 {len(buy_points)} 个买点，{len(sell_points)} 个卖点")
        
        # 3. 计算因子值
        logging.info("开始计算动量因子...")
        factor = SimpleMomentumFactor(window=24)
        factor_values = factor.calculate(data)
        logging.info("因子计算完成")
        
        # 4. 评估因子
        logging.info("开始评估因子...")
        evaluator = FactorEvaluator()
        returns = data['close'].pct_change()  # 计算收益率
        metrics = evaluator.evaluate(factor_values, returns)
        
        # 5. 输出评估结果
        logging.info("因子评估结果：")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
            
        return True
        
    except Exception as e:
        logging.error(f"测试过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    test_factor_framework()