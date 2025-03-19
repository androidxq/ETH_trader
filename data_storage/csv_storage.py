from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

class CSVStorage:
    def __init__(self, base_path: str = None):
        # 如果没有提供路径，使用相对于项目根目录的路径
        if base_path is None:
            # 获取当前文件位置
            current_file = Path(__file__)
            # 项目根目录
            project_root = current_file.parent.parent
            # 默认数据存储路径
            base_path = project_root / "data"
            
        self.base_path = Path(base_path)
        self._init_storage()
        
    def _init_storage(self):
        """初始化存储目录"""
        for dir_name in ["kline", "tick", "depth"]:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)
            
    def save_kline_data_by_month(self, data: pd.DataFrame, symbol: str, interval: str):
        """按月份保存K线数据"""
        if data.empty:
            logging.warning(f"Empty data for {symbol} {interval}")
            return
        
        # 按月份分组
        grouped = data.groupby(data['timestamp'].dt.strftime('%Y%m'))
        
        for month, month_data in grouped:
            filename = f"{symbol}_{interval}_{month}.csv"
            filepath = self.base_path / "kline" / filename
            
            # 如果文件存在，合并数据
            if filepath.exists():
                existing_data = pd.read_csv(filepath)
                existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
                month_data = pd.concat([existing_data, month_data])
                month_data = month_data.drop_duplicates(subset=['timestamp'])
            
            month_data = month_data.sort_values('timestamp')
            month_data.to_csv(filepath, index=False)
            logging.info(f"Saved kline data to {filepath}")