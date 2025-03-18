from data_fetcher.binance_fetcher import BinanceFetcher
from data_storage.csv_storage import CSVStorage
from config.api_config import BINANCE_CONFIG
from config.trading_config import SYMBOL_CONFIG
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_historical_data():
    fetcher = BinanceFetcher()
    storage = CSVStorage()
    
    # 连接API
    fetcher.connect(
        BINANCE_CONFIG["api_key"], 
        BINANCE_CONFIG["api_secret"],
        BINANCE_CONFIG["proxy"]
    )
    
    # 下载所有时间周期的数据
    for interval in SYMBOL_CONFIG["intervals"]:
        logging.info(f"开始下载 {SYMBOL_CONFIG['main_symbol']} {interval} 数据")
        
        data = fetcher.fetch_historical_data(
            symbol=SYMBOL_CONFIG["main_symbol"],
            interval=interval,
            years=2,
            segment_days=30
        )
        
        if not data.empty:
            storage.save_kline_data_by_month(
                data,
                SYMBOL_CONFIG["main_symbol"],
                interval
            )
        else:
            logging.error(f"{interval} 数据下载失败")

if __name__ == "__main__":
    download_historical_data()