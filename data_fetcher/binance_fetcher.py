from binance.client import Client
from datetime import datetime, timedelta, date
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
        """获取U本位合约K线数据，处理API返回数据量限制问题
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            pandas.DataFrame: 指定时间段内的全部K线数据
        """
        all_klines = []
        current_start = start_time
        
        # 如果结束时间未指定，使用当前时间
        if end_time is None:
            end_time = datetime.now()
            
        # 添加日志输出
        logging.info(f"开始获取 {symbol} {interval} 数据: {start_time} - {end_time}")
        
        # 使用循环获取所有数据，处理API每次返回数据量的限制
        while current_start < end_time:
            try:
                # 获取当前开始时间到结束时间的数据
                klines = self.client.futures_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=str(int(current_start.timestamp() * 1000)),
                    end_str=str(int(end_time.timestamp() * 1000)),
                    limit=1000  # 明确指定最大数量
                )
                
                # 如果没有获取到数据，说明这个时间段内没有交易数据
                if not klines:
                    logging.warning(f"时间段内没有数据: {current_start} - {end_time}")
                    break
                    
                # 如果获取到的数据量小于1000，说明数据已经全部获取完成
                if len(klines) < 1000:
                    all_klines.extend(klines)
                    logging.info(f"获取数据完成: {len(klines)} 条记录")
                    break
                    
                # 如果获取到的数据量等于1000，可能还有更多数据
                all_klines.extend(klines)
                
                # 更新开始时间为最后一条记录的时间+1毫秒
                last_time = datetime.fromtimestamp(klines[-1][0]/1000) + timedelta(milliseconds=1)
                
                # 确保时间是向前推进的
                if last_time <= current_start:
                    logging.warning(f"时间没有前进，中止获取: {current_start} vs {last_time}")
                    break
                    
                current_start = last_time
                logging.info(f"已获取 {len(all_klines)} 条记录，继续获取: {current_start} - {end_time}")
                
                # 添加请求间隔，避免触发频率限制
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"获取数据失败: {symbol} {interval} {current_start} - {end_time}, 错误: {e}")
                return pd.DataFrame()
        
        # 转换为DataFrame
        if all_klines:
            return self._convert_to_dataframe(all_klines)
        else:
            return pd.DataFrame()

    def fetch_historical_data(self, symbol: str, interval: str, 
                            start_date=None, end_date=None, 
                            years: int = 2, segment_days: int = 30,
                            progress_callback=None):
        """分段获取历史数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            start_date: 起始日期（date对象）
            end_date: 结束日期（date对象）
            years: 如果未提供start_date，获取多少年的数据
            segment_days: 每段数据的天数
            progress_callback: 进度回调函数，接收(current, total)参数
            
        Returns:
            pandas.DataFrame: 合并后的历史数据
        """
        # 如果提供了精确的日期范围，优先使用它们
        if end_date is None:
            end_time = datetime.now()
        elif isinstance(end_date, date):
            # 转换date对象为当天的结束时间
            end_time = datetime.combine(end_date, datetime.max.time())
        else:
            end_time = end_date

        if start_date is None:
            start_time = end_time - timedelta(days=365 * years)
        elif isinstance(start_date, date):
            # 转换date对象为当天的开始时间
            start_time = datetime.combine(start_date, datetime.min.time())
        else:
            start_time = start_date
        
        all_data = []
        current_start = start_time
        
        # 获取时间间隔的毫秒数
        interval_ms = self._get_interval_ms(interval)
        if interval_ms == 0:
            logging.error(f"无法识别的时间间隔: {interval}")
            return pd.DataFrame()
            
        logging.info(f"下载 {symbol} {interval} 数据，时间范围: {start_time} 到 {end_time}")
        
        # 添加进度条
        total_segments = (end_time - start_time).days // segment_days + 1
        with tqdm(total=total_segments, desc=f"下载 {symbol} {interval} 数据") as pbar:
            segment_count = 0
            while current_start < end_time:
                current_end = min(current_start + timedelta(days=segment_days), end_time)
                
                # 添加重试机制
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        segment_data = self.fetch_kline_data(
                            symbol=symbol,
                            interval=interval,
                            start_time=current_start,
                            end_time=current_end
                        )
                        
                        if not segment_data.empty:
                            all_data.append(segment_data)
                            break
                        else:
                            retry_count += 1
                            time.sleep(1)  # 等待1秒后重试
                    except Exception as e:
                        logging.error(f"获取数据失败: {str(e)}")
                        retry_count += 1
                        time.sleep(1)  # 等待1秒后重试
                
                if retry_count == max_retries:
                    logging.error(f"获取 {current_start} 到 {current_end} 的数据失败，已达到最大重试次数")
                
                current_start = current_end
                segment_count += 1
                pbar.update(1)
                
                # 调用进度回调更新UI
                if progress_callback:
                    progress_callback(segment_count, total_segments)
                
                # 添加请求间隔，避免触发频率限制
                time.sleep(0.5)
        
        if all_data:
            # 合并所有数据
            combined_data = pd.concat(all_data)
            
            # 删除重复数据
            combined_data = combined_data.drop_duplicates(subset=['timestamp'])
            
            # 按时间排序
            combined_data = combined_data.sort_values('timestamp')
            
            # 检查数据连贯性并修复缺失部分
            combined_data = self._check_and_fix_data_continuity(
                combined_data, symbol, interval, start_time, end_time, interval_ms)
            
            # 打印数据统计信息
            logging.info(f"数据下载完成，共 {len(combined_data)} 条记录")
            logging.info(f"时间范围: {combined_data['timestamp'].min()} 到 {combined_data['timestamp'].max()}")
            
            return combined_data
        else:
            logging.error("未获取到任何数据")
            return pd.DataFrame()
            
    def _check_and_fix_data_continuity(self, data, symbol, interval, start_time, end_time, interval_ms):
        """检查数据连贯性，修复缺失的数据
        
        Args:
            data: 已下载的数据
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            interval_ms: 时间间隔的毫秒数
            
        Returns:
            pandas.DataFrame: 修复后的数据
        """
        if data.empty:
            return data
            
        logging.info("检查数据连贯性...")
        
        # 计算相邻时间戳的差值
        data = data.sort_values('timestamp')
        data['next_timestamp'] = data['timestamp'].shift(-1)
        data['time_diff'] = (data['next_timestamp'] - data['timestamp']).dt.total_seconds() * 1000
        
        # 找出缺失的区间（时间差大于间隔的1.5倍，允许有些许误差）
        missing_intervals = data[data['time_diff'] > interval_ms * 1.5].copy()
        
        if missing_intervals.empty:
            logging.info("数据连贯性检查完成，未发现缺失数据")
            return data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        logging.warning(f"发现 {len(missing_intervals)} 个数据缺失区间，正在修复...")
        
        # 保存原始数据
        fixed_data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # 修复每个缺失区间
        for idx, row in missing_intervals.iterrows():
            missing_start = row['timestamp']
            missing_end = row['next_timestamp']
            
            logging.warning(f"修复缺失区间: {missing_start} 到 {missing_end}")
            
            # 尝试重新下载这段数据
            missing_data = self.fetch_kline_data(
                symbol=symbol,
                interval=interval,
                start_time=missing_start + timedelta(milliseconds=interval_ms),  # 避免重复
                end_time=missing_end
            )
            
            if not missing_data.empty:
                # 合并新下载的数据
                fixed_data = pd.concat([fixed_data, missing_data])
                logging.info(f"成功修复缺失区间，添加了 {len(missing_data)} 条记录")
            else:
                logging.warning(f"无法修复缺失区间: {missing_start} 到 {missing_end}")
                
        # 重新排序、去重
        fixed_data = fixed_data.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        
        # 检查起始和结束时间是否完整
        first_timestamp = fixed_data['timestamp'].min()
        last_timestamp = fixed_data['timestamp'].max()
        
        # 检查开始时间
        if first_timestamp > start_time + timedelta(milliseconds=interval_ms):
            logging.warning(f"数据开始时间晚于请求时间: {first_timestamp} > {start_time}")
            early_data = self.fetch_kline_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=first_timestamp
            )
            if not early_data.empty:
                fixed_data = pd.concat([fixed_data, early_data])
                logging.info(f"添加了 {len(early_data)} 条早期数据")
                
        # 检查结束时间
        if last_timestamp < end_time - timedelta(milliseconds=interval_ms):
            logging.warning(f"数据结束时间早于请求时间: {last_timestamp} < {end_time}")
            late_data = self.fetch_kline_data(
                symbol=symbol,
                interval=interval,
                start_time=last_timestamp + timedelta(milliseconds=interval_ms),
                end_time=end_time
            )
            if not late_data.empty:
                fixed_data = pd.concat([fixed_data, late_data])
                logging.info(f"添加了 {len(late_data)} 条后期数据")
                
        # 最终整理
        fixed_data = fixed_data.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        
        # 记录最终修复结果
        total_fixed = len(fixed_data) - len(data)
        if total_fixed > 0:
            logging.info(f"数据连贯性修复完成，共添加了 {total_fixed} 条缺失记录")
        
        return fixed_data
        
    def _get_interval_ms(self, interval):
        """获取时间间隔的毫秒数
        
        Args:
            interval: 时间间隔字符串，如1m, 5m, 1h, 1d等
            
        Returns:
            int: 时间间隔的毫秒数
        """
        # 解析时间间隔
        interval_map = {
            # 分钟
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            # 小时
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            # 天
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            # 周
            '1w': 7 * 24 * 60 * 60 * 1000,
            # 月
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        
        return interval_map.get(interval, 0)

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