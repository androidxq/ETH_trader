"""数据加载模块"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import timedelta
from typing import Optional, Dict, List
from collections import OrderedDict

logger = logging.getLogger(__name__)

class DataCache:
    def __init__(self, max_cache_size: int = 50, max_segment_size: int = 5000):
        """初始化数据缓存管理器
        
        Args:
            max_cache_size: 最大缓存段数量，超过此数量将清理最早的数据段
            max_segment_size: 每个数据段的最大记录数
        """
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.max_segment_size = max_segment_size
        self.segment_info = {}

    def get_segment(self, key: str, start_date=None, end_date=None) -> Optional[pd.DataFrame]:
        """获取指定时间范围的数据段"""
        segment_key = f"{key}_{start_date}_{end_date}"
        if segment_key in self.cache:
            # 更新访问顺序
            value = self.cache.pop(segment_key)
            self.cache[segment_key] = value
            return value
        return None

    def set_segment(self, key: str, value: pd.DataFrame, start_date=None, end_date=None):
        """设置数据段缓存"""
        segment_key = f"{key}_{start_date}_{end_date}"
        if segment_key in self.cache:
            self.cache.pop(segment_key)
        elif len(self.cache) >= self.max_cache_size:
            # 删除最早的数据段
            self.cache.popitem(last=False)
        
        # 记录数据段信息
        self.segment_info[segment_key] = {
            'key': key,
            'start_date': start_date,
            'end_date': end_date,
            'size': len(value)
        }
        
        # 如果数据段过大，进行分割
        if len(value) > self.max_segment_size:
            segments = [value[i:i + self.max_segment_size] 
                       for i in range(0, len(value), self.max_segment_size)]
            for i, segment in enumerate(segments):
                sub_key = f"{segment_key}_part{i}"
                self.cache[sub_key] = segment
        else:
            self.cache[segment_key] = value

    def clear_old_segments(self, key: str):
        """清理指定key的旧数据段"""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(key)]
        for k in keys_to_remove:
            self.cache.pop(k)
            self.segment_info.pop(k, None)

# 全局数据缓存实例
_data_cache = DataCache()

def load_data_segment(data_pattern="data/kline/ETHUSDT_5m_*.csv", start_date=None, end_date=None, buffer_days=2, max_records=5000):
    """
    加载指定时间范围内的数据段，支持按需加载和预加载缓冲区
    
    Args:
        data_pattern: 数据文件glob模式
        start_date: 起始日期(datetime或str)，None表示不限制
        end_date: 结束日期(datetime或str)，None表示不限制
        buffer_days: 预加载缓冲区天数
        max_records: 每个数据段的最大记录数
        
    Returns:
        pandas.DataFrame: 指定时间范围内的数据段
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 获取所有匹配的数据文件
    data_files = sorted(list(project_root.glob(data_pattern)))
    
    if not data_files:
        logger.error(f"未找到匹配 {data_pattern} 的数据文件")
        return pd.DataFrame()

    # 分批加载并处理数据
    chunk_size = min(10000, max_records)  # 每次处理的数据量
    all_data = []
    total_records = 0
    failed_files = []
    
    for file_path in tqdm(data_files, desc="加载数据文件"):
        try:
            # 使用分块读取CSV文件
            file_data = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                if 'timestamp' in chunk.columns:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                    file_data.append(chunk)
                    total_records += len(chunk)
                    
                    # 如果达到最大记录数限制，停止加载
                    if total_records >= max_records:
                        logger.info(f"达到数据段大小限制 {max_records}，停止加载")
                        break
            
            if file_data:
                all_data.extend(file_data)
                logger.info(f"成功加载文件 {file_path.name}，添加 {len(file_data)} 条记录")
            else:
                logger.warning(f"文件 {file_path.name} 没有数据")
                            
        except Exception as e:
            logger.error(f"加载文件 {file_path.name} 失败: {str(e)}")
            failed_files.append((file_path.name, str(e)))
            continue
            
        # 如果达到最大记录数限制，停止加载更多文件
        if total_records >= max_records:
            break
    
    if not all_data:
        if failed_files:
            logger.error(f"所有文件加载失败，失败详情：")
            for file_name, error in failed_files:
                logger.error(f"  - {file_name}: {error}")
        else:
            logger.error("没有找到数据")
        return pd.DataFrame()
    
    # 合并并排序去重
    data = pd.concat(all_data)
    data = data.sort_values('timestamp').drop_duplicates()
    
    # 如果数据量超过限制，只保留最新的数据
    if len(data) > max_records:
        data = data.tail(max_records)
        logger.info(f"数据量超过限制，已截取最新的{max_records}条记录")
    
    logger.info(f"数据段加载完成，共 {len(data)} 条记录")
    if 'timestamp' in data.columns:
        logger.info(f"数据段时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    return data

def load_data_files(data_pattern="data/kline/ETHUSDT_5m_*.csv", start_date=None, end_date=None, buffer_days=2, max_records=10000):
    """
    加载指定时间范围内的数据文件，支持分段加载
    
    Args:
        data_pattern: 数据文件glob模式
        start_date: 起始日期(datetime或str)，None表示不限制
        end_date: 结束日期(datetime或str)，None表示不限制
        buffer_days: 预加载缓冲区天数
        max_records: 每个数据段的最大记录数，默认10000条
        
    Returns:
        pandas.DataFrame: 合并后的数据
    """
    # 如果没有指定时间范围，默认加载最近的数据
    if not end_date:
        end_date = pd.Timestamp.now()
    if not start_date:
        start_date = pd.to_datetime(end_date) - timedelta(days=7)  # 默认只加载7天数据
    
    # 转换时间格式
    try:
        end_date = pd.to_datetime(end_date)
        start_date = pd.to_datetime(start_date)
    except Exception as e:
        logger.error(f"时间格式转换失败: {str(e)}")
        return pd.DataFrame()
    
    # 严格限制时间范围和数据量
    date_range = (end_date - start_date).days
    if date_range > 365:  # 最多加载1年数据
        start_date = end_date - timedelta(days=365)
        logger.warning(f"时间范围过大，已限制为1年: {start_date} -> {end_date}")
    
    # 加载数据段
    data = load_data_segment(data_pattern, start_date, end_date, buffer_days, max_records)
    
    # 严格限制数据量
    if len(data) > max_records:
        data = data.tail(max_records)
        logger.info(f"数据量超过限制，已截取最新的{max_records}条记录")
    
    # 清理旧的缓存数据段
    _data_cache.clear_old_segments(data_pattern)
    
    return data