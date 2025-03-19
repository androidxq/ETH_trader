from data_fetcher.binance_fetcher import BinanceFetcher
from data_storage.csv_storage import CSVStorage
from config.api_config import BINANCE_CONFIG
from config.trading_config import SYMBOL_CONFIG
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

def test_fetch_and_store():
    fetcher = BinanceFetcher()
    storage = CSVStorage()  # 使用默认相对路径
    
    fetcher.connect(
        BINANCE_CONFIG["api_key"], 
        BINANCE_CONFIG["api_secret"],
        BINANCE_CONFIG["proxy"]
    )
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)
    
    data = fetcher.fetch_kline_data(
        symbol=SYMBOL_CONFIG["main_symbol"],
        interval=SYMBOL_CONFIG["intervals"][0],  # 使用1分钟数据
        start_time=start_time,
        end_time=end_time
    )
    
    storage.save_kline_data_by_month(
        data, 
        SYMBOL_CONFIG["main_symbol"], 
        SYMBOL_CONFIG["intervals"][0]
    )

if __name__ == "__main__":
    test_fetch_and_store()