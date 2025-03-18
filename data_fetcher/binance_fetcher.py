from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import logging
import time
from tqdm import tqdm

class BinanceFetcher:
    def __init__(self):
        self.client = None
        
    def connect(self, key: str, secret: str, proxy: str = None):
        """连接币安API"""
        kwargs = {}
        if proxy:
            kwargs['proxies'] = {
                'http': proxy,
                'https': proxy
            }
        self.client = Client(key, secret, requests_params=kwargs)
        
    def fetch_kline_data(self, symbol: str, interval: str, 
                        start_time: datetime, end_time: datetime = None):
        """获取U本位合约K线数据"""
        try:
            klines = self.client.futures_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(int(start_time.timestamp() * 1000)),
                end_str=str(int(end_time.timestamp() * 1000)) if end_time else None,
            )
            # 添加请求间隔，避免触发频率限制
            time.sleep(0.5)
            return self._convert_to_dataframe(klines)
        except Exception as e:
            logging.error(f"获取数据失败: {symbol} {interval} {start_time} - {end_time}, 错误: {e}")
            return pd.DataFrame()

    def fetch_historical_data(self, symbol: str, interval: str, 
                            years: int = 2, segment_days: int = 30):
        """分段获取历史数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365 * years)
        
        all_data = []
        current_start = start_time
        
        # 添加进度条
        total_segments = (end_time - start_time).days // segment_days + 1
        with tqdm(total=total_segments, desc=f"下载 {symbol} {interval} 数据") as pbar:
            while current_start < end_time:
                current_end = min(current_start + timedelta(days=segment_days), end_time)
                
                segment_data = self.fetch_kline_data(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=current_end
                )
                
                if not segment_data.empty:
                    all_data.append(segment_data)
                
                current_start = current_end
                pbar.update(1)
        
        if all_data:
            return pd.concat(all_data).drop_duplicates().sort_values('timestamp')
        return pd.DataFrame()

    def _convert_to_dataframe(self, klines: list) -> pd.DataFrame:
        """转换数据为DataFrame格式"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 转换数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]