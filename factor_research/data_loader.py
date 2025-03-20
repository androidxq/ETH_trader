"""
数据加载模块
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def load_data_files(data_pattern="data/kline/ETHUSDT_5m_*.csv"):
    """
    加载所有匹配模式的数据文件并合并
    
    Args:
        data_pattern: 数据文件glob模式，默认加载所有5分钟K线数据
        
    Returns:
        pandas.DataFrame: 合并后的数据
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 获取所有匹配的数据文件
    data_files = list(project_root.glob(data_pattern))
    
    if not data_files:
        logger.error(f"未找到匹配 {data_pattern} 的数据文件")
        return pd.DataFrame()
        
    logger.info(f"找到 {len(data_files)} 个数据文件")
    
    # 合并所有文件数据
    all_data = []
    for file_path in tqdm(data_files, desc="加载数据文件"):
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_data.append(df)
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {str(e)}")
    
    if not all_data:
        logger.error("所有文件加载失败")
        return pd.DataFrame()
    
    # 合并并排序去重
    data = pd.concat(all_data)
    data = data.sort_values('timestamp').drop_duplicates()
    
    logger.info(f"数据加载完成，共 {len(data)} 条记录")
    if 'timestamp' in data.columns:
        logger.info(f"数据时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    return data 