from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QSpinBox, QPushButton, QProgressBar,
                             QCheckBox, QGroupBox, QMessageBox, QDateEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
from data_fetcher.binance_fetcher import BinanceFetcher
from data_storage.csv_storage import CSVStorage
from config.api_config import BINANCE_CONFIG
import logging

class DownloadWorker(QThread):
    """下载工作线程"""
    progress = pyqtSignal(int, int)  # 当前进度，总进度
    segment_progress = pyqtSignal(int, int)  # 当前段进度，总段数
    status = pyqtSignal(str)  # 状态信息
    finished = pyqtSignal()  # 完成信号
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, symbol, intervals, start_date, end_date, segment_days):
        super().__init__()
        self.symbol = symbol
        self.intervals = intervals
        self.start_date = start_date
        self.end_date = end_date
        self.segment_days = segment_days
        self.is_running = True
        
    def run(self):
        try:
            fetcher = BinanceFetcher()
            storage = CSVStorage()
            
            # 连接API
            fetcher.connect(
                BINANCE_CONFIG["api_key"],
                BINANCE_CONFIG["api_secret"],
                BINANCE_CONFIG["proxy"]
            )
            
            total_intervals = len(self.intervals)
            for i, interval in enumerate(self.intervals):
                if not self.is_running:
                    self.status.emit("下载已暂停")
                    return
                    
                self.status.emit(f"正在下载 {self.symbol} {interval} 数据...")
                
                # 创建段下载进度回调函数
                def update_segment_progress(current, total):
                    self.segment_progress.emit(current, total)
                    self.status.emit(f"正在下载 {self.symbol} {interval} 数据 - 段 {current}/{total}")
                
                data = fetcher.fetch_historical_data(
                    symbol=self.symbol,
                    interval=interval,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    segment_days=self.segment_days,
                    progress_callback=update_segment_progress
                )
                
                if not data.empty:
                    storage.save_kline_data_by_month(
                        data,
                        self.symbol,
                        interval
                    )
                    self.status.emit(f"已保存 {len(data)} 条 {interval} 数据")
                else:
                    self.error.emit(f"{interval} 数据下载失败")
                    
                # 更新总进度
                self.progress.emit(i + 1, total_intervals)
                
            self.status.emit("下载完成")
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"下载出错: {str(e)}")
            
    def stop(self):
        self.is_running = False

class DataDownloadWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.download_worker = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 交易对选择
        symbol_group = QGroupBox("交易对设置")
        symbol_layout = QHBoxLayout()
        symbol_label = QLabel("交易对:")
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["ETHUSDT", "BTCUSDT", "BNBUSDT"])
        symbol_layout.addWidget(symbol_label)
        symbol_layout.addWidget(self.symbol_combo)
        symbol_group.setLayout(symbol_layout)
        layout.addWidget(symbol_group)
        
        # 时间周期选择
        interval_group = QGroupBox("时间周期设置")
        interval_layout = QVBoxLayout()
        self.interval_checkboxes = {}
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for interval in intervals:
            checkbox = QCheckBox(interval)
            self.interval_checkboxes[interval] = checkbox
            interval_layout.addWidget(checkbox)
        interval_group.setLayout(interval_layout)
        layout.addWidget(interval_group)
        
        # 时间范围设置 - 修改为日历选择
        time_group = QGroupBox("时间范围设置")
        time_layout = QVBoxLayout()
        
        # 开始日期选择器
        start_date_layout = QHBoxLayout()
        start_date_label = QLabel("开始日期:")
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setCalendarPopup(True)  # 允许弹出日历
        self.start_date_edit.setDate(QDate.currentDate().addYears(-2))  # 默认2年前
        start_date_layout.addWidget(start_date_label)
        start_date_layout.addWidget(self.start_date_edit)
        time_layout.addLayout(start_date_layout)
        
        # 结束日期选择器
        end_date_layout = QHBoxLayout()
        end_date_label = QLabel("结束日期:")
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setCalendarPopup(True)  # 允许弹出日历
        self.end_date_edit.setDate(QDate.currentDate())  # 默认当前日期
        end_date_layout.addWidget(end_date_label)
        end_date_layout.addWidget(self.end_date_edit)
        time_layout.addLayout(end_date_layout)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # 进度显示
        progress_group = QGroupBox("下载进度")
        progress_layout = QVBoxLayout()
        
        # 总进度条
        total_progress_layout = QHBoxLayout()
        total_progress_label = QLabel("总进度:")
        self.progress_bar = QProgressBar()
        total_progress_layout.addWidget(total_progress_label)
        total_progress_layout.addWidget(self.progress_bar)
        progress_layout.addLayout(total_progress_layout)
        
        # 段进度条
        segment_progress_layout = QHBoxLayout()
        segment_progress_label = QLabel("当前段进度:")
        self.segment_progress_bar = QProgressBar()
        segment_progress_layout.addWidget(segment_progress_label)
        segment_progress_layout.addWidget(self.segment_progress_bar)
        progress_layout.addLayout(segment_progress_layout)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始下载")
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 连接信号
        self.start_button.clicked.connect(self.start_download)
        self.pause_button.clicked.connect(self.pause_download)
    
    def start_download(self):
        # 获取选中的时间周期
        selected_intervals = [
            interval for interval, checkbox in self.interval_checkboxes.items()
            if checkbox.isChecked()
        ]
        
        if not selected_intervals:
            QMessageBox.warning(self, "警告", "请至少选择一个时间周期")
            return
            
        # 获取日期范围
        start_date = self.start_date_edit.date().toPyDate()
        end_date = self.end_date_edit.date().toPyDate()
        
        # 验证日期范围
        if start_date >= end_date:
            QMessageBox.warning(self, "警告", "开始日期必须早于结束日期")
            return
            
        # 禁用开始按钮，启用暂停按钮
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        self.segment_progress_bar.setValue(0)
        self.status_label.setText("准备下载...")
        
        # 创建并启动下载线程
        self.download_worker = DownloadWorker(
            self.symbol_combo.currentText(),
            selected_intervals,
            start_date,
            end_date,
            30  # 固定30天为一个分段
        )
        
        # 连接信号
        self.download_worker.progress.connect(self.update_progress)
        self.download_worker.segment_progress.connect(self.update_segment_progress)
        self.download_worker.status.connect(self.update_status)
        self.download_worker.finished.connect(self.download_finished)
        self.download_worker.error.connect(self.download_error)
        
        # 启动下载
        self.download_worker.start()
        
    def pause_download(self):
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.stop()
            self.pause_button.setText("继续")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.resume_download)
            
    def resume_download(self):
        if self.download_worker:
            self.download_worker.is_running = True
            self.download_worker.start()
            self.pause_button.setText("暂停")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_download)
            
    def update_progress(self, current, total):
        """更新总进度条"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} ({int(current/total*100)}%)")
        
    def update_segment_progress(self, current, total):
        """更新段进度条"""
        self.segment_progress_bar.setMaximum(total)
        self.segment_progress_bar.setValue(current)
        self.segment_progress_bar.setFormat(f"{current}/{total} ({int(current/total*100)}%)")
        
    def update_status(self, message):
        self.status_label.setText(message)
        
    def download_finished(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        QMessageBox.information(self, "完成", "数据下载完成")
        
    def download_error(self, error_message):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        QMessageBox.critical(self, "错误", error_message) 