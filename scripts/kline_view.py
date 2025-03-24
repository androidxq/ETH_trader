"""
K线图显示模块

提供TradingView风格的K线图显示，支持多币种、多时间周期切换
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mplfinance as mpf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入PyQt6
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QPushButton, QGroupBox, QSplitter, 
                           QFrame, QGridLayout, QSizePolicy, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QSize

# 导入项目相关模块
from factor_research.data_loader import load_data_files
from config.trading_config import SYMBOL_CONFIG

# 定义时间周期映射，用于显示名称
INTERVAL_DISPLAY = {
    "1m": "1分钟",
    "3m": "3分钟",
    "5m": "5分钟", 
    "15m": "15分钟",
    "30m": "30分钟",
    "1h": "1小时",
    "2h": "2小时",
    "4h": "4小时",
    "6h": "6小时",
    "8h": "8小时", 
    "12h": "12小时",
    "1d": "日线"
}

class KlineFigureCanvas(FigureCanvas):
    """自定义的K线图Canvas"""
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # 创建子图用于成交量
        self.volume_axes = self.axes.twinx()
        # 调整成交量区域，减小高度，防止遮挡K线
        self.volume_axes.set_position([0.1, 0.1, 0.8, 0.1])  # 高度从0.15进一步减小到0.1
        self.axes.set_position([0.1, 0.25, 0.8, 0.7])  # 增加主K线图高度，从0.6增加到0.7
        
        # K线显示设置
        self.bar_width = 0.8  # 默认K线宽度
        self.min_bar_width = 0.3  # 最小K线宽度
        self.max_bar_width = 1.5  # 最大K线宽度
        
        # 时间轴导航参数
        self.time_offset = 0  # 时间偏移量，0表示最新数据
        
        # 初始化基类
        super(KlineFigureCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 设置样式
        plt.style.use('dark_background')
        self.fig.patch.set_facecolor('#1e1e1e')
        self.axes.set_facecolor('#1e1e1e')
        self.volume_axes.set_facecolor('#1e1e1e')
        
        # 隐藏成交量y轴
        self.volume_axes.get_yaxis().set_visible(False)
        
        # 为界面适应性调整
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # 连接鼠标滚轮事件 - 修复：确保事件连接正确
        self.mpl_connect('scroll_event', self.on_scroll)
        print("已连接mpl鼠标滚轮事件")  # 添加调试信息
        
        # 存储当前数据，以便重新绘制
        self.current_data = None
        self.current_title = ""
        self.full_data = None  # 存储完整数据集
        self.date_range = "1d"  # 当前显示的时间范围
        
        # 设置允许鼠标事件
        self.setMouseTracking(True)
        
    def move_time_window(self, direction):
        """移动时间窗口，查看前后时间段的K线
        
        Args:
            direction: 移动方向，1表示向前（查看更早数据），-1表示向后（查看更新数据）
        """
        if self.full_data is None or self.full_data.empty:
            return
            
        # 根据当前时间范围确定移动步长
        if self.date_range == "1d":
            step = timedelta(hours=6)  # 移动6小时
        elif self.date_range == "3d":
            step = timedelta(days=1)   # 移动1天
        elif self.date_range == "1w":
            step = timedelta(days=2)   # 移动2天
        elif self.date_range == "2w":
            step = timedelta(days=3)   # 移动3天
        elif self.date_range == "1m":
            step = timedelta(days=7)   # 移动7天
        elif self.date_range == "3m":
            step = timedelta(days=14)  # 移动14天
        else:  # "all"
            return  # 全部数据时不需要移动
            
        # 更新时间偏移
        self.time_offset += direction
        
        # 限制偏移范围，不超过数据集大小
        max_offset = len(self.full_data) // 10  # 假设一个合理的最大偏移
        self.time_offset = max(-max_offset, min(self.time_offset, 0))
        
        # 计算新的时间范围
        if self.time_offset == 0:
            # 重置为默认范围（最新数据）
            self.plot_kline_with_range(self.full_data, self.current_title, self.date_range)
        else:
            # 计算新的时间窗口
            data_copy = self.full_data.copy()
            latest_time = data_copy['timestamp'].max() + step * self.time_offset
            
            # 根据日期范围设置时间窗口
            if self.date_range == "1d":
                start_time = latest_time - timedelta(days=1)
            elif self.date_range == "3d":
                start_time = latest_time - timedelta(days=3)
            elif self.date_range == "1w":
                start_time = latest_time - timedelta(days=7)
            elif self.date_range == "2w":
                start_time = latest_time - timedelta(days=14)
            elif self.date_range == "1m":
                start_time = latest_time - timedelta(days=30)
            elif self.date_range == "3m":
                start_time = latest_time - timedelta(days=90)
            else:  # "all"
                return data_copy
                
            # 筛选数据
            filtered_data = data_copy[(data_copy['timestamp'] >= start_time) & 
                                     (data_copy['timestamp'] <= latest_time)]
            
            # 绘制新的时间范围
            if not filtered_data.empty:
                title = f"{self.current_title} (偏移: {self.time_offset})"
                self.plot_kline(filtered_data, title)
        
    def wheelEvent(self, event):
        """重写Qt的wheelEvent方法以捕获滚轮事件"""
        print("捕获到Qt滚轮事件")
        delta = event.angleDelta().y()
        
        old_width = self.bar_width
        
        if delta > 0:  # 向上滚动
            self.bar_width = min(self.bar_width + 0.1, self.max_bar_width)
            print(f"Qt事件：增大K线宽度为：{self.bar_width}")
        else:  # 向下滚动
            self.bar_width = max(self.bar_width - 0.1, self.min_bar_width)
            print(f"Qt事件：减小K线宽度为：{self.bar_width}")
            
        # 如果宽度真的变化了才重绘
        if abs(old_width - self.bar_width) > 0.01 and self.current_data is not None:
            print("Qt事件：重新绘制K线图")
            self.plot_kline(self.current_data, self.current_title)
            
            # 通知父窗口更新宽度标签
            if self.parent() and hasattr(self.parent(), 'update_width_label'):
                self.parent().update_width_label()
            
            # 强制刷新UI - 使用Qt的更新方法确保视图更新
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.update()
            QApplication.processEvents()  # 强制处理所有待处理的Qt事件
        
        # 接受事件，避免传递给父窗口
        event.accept()
    
    def on_scroll(self, event):
        """处理matplotlib的鼠标滚轮事件，缩放K线大小"""
        print(f"捕获到matplotlib滚轮事件：{event.button}，当前宽度：{self.bar_width}")  # 添加调试信息
        
        old_width = self.bar_width
        
        if event.button == 'up':  # 鼠标滚轮向上滚动，增加K线宽度
            self.bar_width = min(self.bar_width + 0.1, self.max_bar_width)
            print(f"增大K线宽度为：{self.bar_width}")
        elif event.button == 'down':  # 鼠标滚轮向下滚动，减小K线宽度
            self.bar_width = max(self.bar_width - 0.1, self.min_bar_width)
            print(f"减小K线宽度为：{self.bar_width}")
            
        # 如果宽度真的变化了才重绘
        if abs(old_width - self.bar_width) > 0.01 and self.current_data is not None:
            print("重新绘制K线图")
            self.plot_kline(self.current_data, self.current_title)
            
            # 通知父窗口更新宽度标签
            if self.parent() and hasattr(self.parent(), 'update_width_label'):
                self.parent().update_width_label()
            
            # 强制刷新UI
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.update()
            QApplication.processEvents()  # 强制处理所有待处理的Qt事件
    
    def plot_kline_with_range(self, data, title="", date_range="1d"):
        """根据时间范围绘制K线图
        
        Args:
            data: pandas DataFrame，包含完整OHLCV数据
            title: 图表标题
            date_range: 范围字符串("1d", "3d", "1w", "2w", "1m", "3m", "all")
        """
        # 存储完整数据和日期范围，以便时间导航
        self.full_data = data
        self.date_range = date_range
        self.time_offset = 0  # 重置时间偏移
        
        # 根据时间范围筛选数据
        if date_range == "all" or data.empty:
            filtered_data = data
        else:
            # 确保数据已排序
            data_copy = data.copy().sort_values('timestamp')
            
            # 获取当前最新数据的时间
            latest_time = data_copy['timestamp'].max()
            
            # 根据范围设置开始时间
            if date_range == "1d":
                start_time = latest_time - timedelta(days=1)
            elif date_range == "3d":
                start_time = latest_time - timedelta(days=3)
            elif date_range == "1w":
                start_time = latest_time - timedelta(days=7)
            elif date_range == "2w":
                start_time = latest_time - timedelta(days=14)
            elif date_range == "1m":
                start_time = latest_time - timedelta(days=30)
            elif date_range == "3m":
                start_time = latest_time - timedelta(days=90)
            
            filtered_data = data_copy[data_copy['timestamp'] >= start_time]
        
        # 绘制筛选后的数据
        self.plot_kline(filtered_data, title)
    
    def plot_kline(self, data, title=""):
        """绘制K线图
        
        Args:
            data: pandas DataFrame，包含OHLCV数据
            title: 图表标题
        """
        # 存储当前数据，以便在缩放时重新绘制
        self.current_data = data
        self.current_title = title
        
        # 清除当前图表
        self.axes.clear()
        self.volume_axes.clear()
        
        # 如果数据为空则返回
        if data.empty:
            self.draw()
            return

        # 确保数据格式正确
        if 'timestamp' in data.columns:
            data = data.copy()
            data.set_index('timestamp', inplace=True)
        
        try:
            # 设置TradingView风格
            mc = mpf.make_marketcolors(
                up='#26a69a', down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume={'up': '#26a69a', 'down': '#ef5350'}
            )
            
            s = mpf.make_mpf_style(
                base_mpf_style='nightclouds',
                marketcolors=mc,
                gridstyle=':',
                gridcolor='#323232',
                gridaxis='both',
                facecolor='#1e1e1e',
                figcolor='#1e1e1e',
                y_on_right=True
            )
            
            # 修复: 不创建新的Figure，直接在现有axes上绘制
            mpf.plot(
                data,
                type='candle',
                style=s,
                ax=self.axes,
                volume=self.volume_axes,
                # 删除title参数，直接设置轴标题
                ylabel='价格',
                ylabel_lower='成交量',
                datetime_format='%Y-%m-%d %H:%M',
                show_nontrading=False
            )
            
            # 手动设置标题
            self.axes.set_title(title, color='white', fontsize=12)
            
        except Exception as e:
            # 如果mpf.plot失败，使用备用方法绘制
            print(f"使用mpf.plot绘制失败，错误: {e}，使用备用方法")
            
            # 设置样式
            self.axes.set_facecolor('#1e1e1e')
            self.volume_axes.set_facecolor('#1e1e1e')
            
            # 绘制K线图 - 修复索引不匹配和线重叠问题
            # 计算上涨和下跌的K线
            data_copy = data.copy()
            up_mask = data_copy['close'] > data_copy['open']
            down_mask = ~up_mask
            
            # 生成x轴位置，修复：增加步长，避免K线重叠
            x = np.arange(len(data_copy))
            # 使用当前设置的K线宽度
            bar_width = self.bar_width
            
            # 绘制上涨K线
            up_indices = np.where(up_mask)[0]
            if len(up_indices) > 0:
                up_highs = data_copy.iloc[up_indices]['high'].values
                up_lows = data_copy.iloc[up_indices]['low'].values
                up_opens = data_copy.iloc[up_indices]['open'].values
                up_closes = data_copy.iloc[up_indices]['close'].values
                up_x = x[up_indices]
                
                # 绘制影线 - 使用细线
                self.axes.vlines(
                    up_x, up_lows, up_highs, 
                    color='#26a69a', linewidth=1
                )
                
                # 绘制实体 - 修复：使用矩形代替线条，更清晰显示K线实体
                for i in range(len(up_x)):
                    self.axes.bar(
                        up_x[i], 
                        up_closes[i] - up_opens[i],  # 高度是收盘价与开盘价的差
                        bottom=up_opens[i],          # 从开盘价开始
                        color='#26a69a', 
                        width=bar_width,             # 使用当前设置的K线宽度
                        alpha=0.8
                    )
                
            # 绘制下跌K线
            down_indices = np.where(down_mask)[0]
            if len(down_indices) > 0:
                down_highs = data_copy.iloc[down_indices]['high'].values
                down_lows = data_copy.iloc[down_indices]['low'].values
                down_opens = data_copy.iloc[down_indices]['open'].values
                down_closes = data_copy.iloc[down_indices]['close'].values
                down_x = x[down_indices]
                
                # 绘制影线 - 使用细线
                self.axes.vlines(
                    down_x, down_lows, down_highs, 
                    color='#ef5350', linewidth=1
                )
                
                # 绘制实体 - 修复：使用矩形代替线条
                for i in range(len(down_x)):
                    self.axes.bar(
                        down_x[i], 
                        down_closes[i] - down_opens[i],  # 高度是收盘价与开盘价的差（负值）
                        bottom=down_opens[i],            # 从开盘价开始
                        color='#ef5350', 
                        width=bar_width,                 # 使用当前设置的K线宽度
                        alpha=0.8
                    )
            
            # 绘制成交量 - 使用相同的x轴位置
            volume = data_copy['volume'].values
            # 缩小成交量高度，设置为原始数据的40%，从70%减小
            volume = volume * 0.4
            colors = np.where(data_copy['close'].values > data_copy['open'].values, '#26a69a', '#ef5350')
            
            for i in range(len(x)):
                self.volume_axes.bar(
                    x[i], volume[i],
                    color=colors[i],
                    width=bar_width, alpha=0.5
                )
            
            # 设置x轴刻度，显示时间
            date_labels = [d.strftime('%m-%d %H:%M') for d in data_copy.index]
            step = max(1, len(date_labels) // 10)  # 最多显示10个标签
            self.axes.set_xticks(x[::step])
            self.axes.set_xticklabels(date_labels[::step], rotation=45, ha='right')
            
            # 设置标题
            self.axes.set_title(title, color='white')
            
        # 设置网格线和标签格式化
        self.axes.grid(True, linestyle=':', color='#323232', alpha=0.3)
        self.axes.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        
        # 隐藏成交量y轴
        self.volume_axes.get_yaxis().set_visible(False)
        
        # 设置中文字体，解决中文显示警告
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        # 刷新绘图
        self.draw()
        # 强制立即更新
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.update()
        # 添加Qt应用程序事件处理，确保UI响应
        QApplication.processEvents()


class KlineViewWidget(QWidget):
    """K线图显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loaded_data = {}  # 存储已加载的不同币种和时间周期的数据
        self.current_symbol = SYMBOL_CONFIG["main_symbol"]
        self.current_interval = "5m"  # 默认5分钟K线
        self.current_date_range = "1d"  # 默认显示1天数据
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # ===== 控制面板 =====
        control_group = QGroupBox("K线图控制面板")
        control_layout = QGridLayout(control_group)
        control_layout.setContentsMargins(10, 20, 10, 10)
        control_layout.setSpacing(10)
        
        # 币种选择
        symbol_label = QLabel("币种:")
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItem(f"{SYMBOL_CONFIG['main_symbol']} (永续合约)")
        self.symbol_combo.currentIndexChanged.connect(self.on_symbol_changed)
        
        # 时间周期选择
        interval_label = QLabel("时间周期:")
        self.interval_combo = QComboBox()
        for interval, display_name in INTERVAL_DISPLAY.items():
            self.interval_combo.addItem(display_name, interval)
        # 默认选择5分钟
        self.interval_combo.setCurrentText(INTERVAL_DISPLAY["5m"])
        self.interval_combo.currentIndexChanged.connect(self.on_interval_changed)
        
        # 时间范围选择
        range_label = QLabel("时间范围:")
        self.range_combo = QComboBox()
        self.range_combo.addItems(["1天", "3天", "1周", "2周", "1个月", "3个月", "全部"])
        self.range_combo.currentIndexChanged.connect(self.on_range_changed)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.clicked.connect(self.load_and_plot_data)
        
        # 添加控件到网格布局
        control_layout.addWidget(symbol_label, 0, 0)
        control_layout.addWidget(self.symbol_combo, 0, 1)
        control_layout.addWidget(interval_label, 0, 2)
        control_layout.addWidget(self.interval_combo, 0, 3)
        control_layout.addWidget(range_label, 0, 4)
        control_layout.addWidget(self.range_combo, 0, 5)
        control_layout.addWidget(self.refresh_button, 0, 6)
        
        # ===== 时间导航控制面板 =====
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(20)
        
        # 上一时间段按钮
        self.prev_button = QPushButton("< 上一时间段")
        self.prev_button.setMinimumWidth(120)
        self.prev_button.clicked.connect(self.on_prev_time_clicked)
        
        # 当前位置显示
        self.position_label = QLabel("当前：最新数据")
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setStyleSheet("color: #2196F3;")
        
        # 下一时间段按钮
        self.next_button = QPushButton("下一时间段 >")
        self.next_button.setMinimumWidth(120)
        self.next_button.clicked.connect(self.on_next_time_clicked)
        
        # 重置按钮
        self.reset_button = QPushButton("重置到最新")
        self.reset_button.setMinimumWidth(120)
        self.reset_button.clicked.connect(self.on_reset_time_clicked)
        
        # 添加到导航布局
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.position_label)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.reset_button)
        
        # ===== K线图显示区域 =====
        self.kline_canvas = KlineFigureCanvas(self, width=10, height=8)
        # 添加：确保鼠标事件传递给Canvas
        self.kline_canvas.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.kline_canvas.setAcceptDrops(True)
        
        # ===== 信息面板 =====
        info_group = QGroupBox("价格信息")
        info_layout = QHBoxLayout(info_group)
        info_layout.setContentsMargins(10, 20, 10, 10)
        
        # 创建价格标签
        self.price_labels = {}
        for label_name in ["开盘价", "最高价", "最低价", "收盘价", "成交量"]:
            label = QLabel(f"{label_name}: -")
            info_layout.addWidget(label)
            self.price_labels[label_name] = label
            
        # 添加K线宽度显示
        self.width_label = QLabel("K线宽度: 0.8")
        info_layout.addWidget(self.width_label)
        
        # 添加滚轮提示
        scroll_tip = QLabel("提示: 使用鼠标滚轮调整K线宽度")
        scroll_tip.setStyleSheet("color: #2196F3;")
        info_layout.addWidget(scroll_tip)
        
        # 添加到主布局
        main_layout.addWidget(control_group)
        main_layout.addLayout(nav_layout)  # 添加时间导航控制面板
        main_layout.addWidget(self.kline_canvas, 1)  # 1表示伸展因子
        main_layout.addWidget(info_group)
        
        # 设置鼠标追踪，允许接收所有鼠标事件
        self.setMouseTracking(True)
        
        # 初始加载数据
        self.load_and_plot_data()
    
    def on_prev_time_clicked(self):
        """查看前一时间段"""
        self.kline_canvas.move_time_window(1)  # 向前移动
        self.update_position_label()
    
    def on_next_time_clicked(self):
        """查看后一时间段"""
        self.kline_canvas.move_time_window(-1)  # 向后移动
        self.update_position_label()
    
    def on_reset_time_clicked(self):
        """重置到最新数据"""
        if self.kline_canvas.time_offset != 0:
            self.kline_canvas.time_offset = 0
            self.load_and_plot_data()  # 重新加载当前数据
            self.update_position_label()
    
    def update_position_label(self):
        """更新位置标签"""
        offset = self.kline_canvas.time_offset
        if offset == 0:
            self.position_label.setText("当前：最新数据")
        else:
            self.position_label.setText(f"当前：偏移 {offset} 个时间单位")
    
    def load_data(self, symbol, interval):
        """加载特定币种和时间周期的数据
        
        Args:
            symbol: 币种代码，如ETHUSDT
            interval: 时间周期，如5m
            
        Returns:
            pandas DataFrame: 加载的数据
        """
        # 检查是否已经加载过
        key = f"{symbol}_{interval}"
        if key in self.loaded_data:
            return self.loaded_data[key]
            
        # 加载数据
        data_pattern = f"data/kline/{symbol}_{interval}_*.csv"
        data = load_data_files(data_pattern)
        
        # 如果数据为空，则返回空DataFrame
        if data.empty:
            print(f"未找到数据: {data_pattern}")
            return pd.DataFrame()
            
        # 缓存数据
        self.loaded_data[key] = data
        return data
    
    def filter_by_date_range(self, data, date_range):
        """根据时间范围筛选数据
        
        Args:
            data: pandas DataFrame，包含时间索引
            date_range: 范围字符串("1d", "3d", "1w", "2w", "1m", "3m", "all")
            
        Returns:
            pandas DataFrame: 筛选后的数据
        """
        if data.empty:
            return data
            
        # 确保数据已排序
        data = data.sort_values('timestamp')
        
        # 获取当前最新数据的时间
        latest_time = data['timestamp'].max()
        
        # 根据范围设置开始时间
        if date_range == "1d":
            start_time = latest_time - timedelta(days=1)
        elif date_range == "3d":
            start_time = latest_time - timedelta(days=3)
        elif date_range == "1w":
            start_time = latest_time - timedelta(days=7)
        elif date_range == "2w":
            start_time = latest_time - timedelta(days=14)
        elif date_range == "1m":
            start_time = latest_time - timedelta(days=30)
        elif date_range == "3m":
            start_time = latest_time - timedelta(days=90)
        else:  # "all"
            return data
            
        return data[data['timestamp'] >= start_time]
    
    def load_and_plot_data(self):
        """加载并绘制当前选择的K线数据"""
        # 获取当前选择
        symbol = self.current_symbol
        interval = self.current_interval
        date_range_mapping = {
            0: "1d", 1: "3d", 2: "1w", 3: "2w", 
            4: "1m", 5: "3m", 6: "all"
        }
        date_range = date_range_mapping[self.range_combo.currentIndex()]
        
        # 加载数据
        data = self.load_data(symbol, interval)
        if data.empty:
            self.update_price_info(None)
            return
            
        # 设置图表标题
        title = f"{symbol} {INTERVAL_DISPLAY.get(interval, interval)}"
        
        # 使用新的带范围绘制方法
        self.kline_canvas.plot_kline_with_range(data, title, date_range)
        
        # 如果有数据，更新最新价格信息
        latest_data = data.sort_values('timestamp').iloc[-1]
        self.update_price_info(latest_data)
        
        # 更新K线宽度显示
        self.update_width_label()
        
        # 更新位置标签
        self.update_position_label()
    
    def update_price_info(self, latest_row):
        """更新价格信息面板
        
        Args:
            latest_row: DataFrame中的最新数据行
        """
        if latest_row is None:
            # 清空价格信息
            for label_name in self.price_labels:
                self.price_labels[label_name].setText(f"{label_name}: -")
            return
            
        # 更新各项数据
        self.price_labels["开盘价"].setText(f"开盘价: {latest_row['open']:.2f}")
        self.price_labels["最高价"].setText(f"最高价: {latest_row['high']:.2f}")
        self.price_labels["最低价"].setText(f"最低价: {latest_row['low']:.2f}")
        self.price_labels["收盘价"].setText(f"收盘价: {latest_row['close']:.2f}")
        self.price_labels["成交量"].setText(f"成交量: {latest_row['volume']:.2f}")
    
    def on_symbol_changed(self, index):
        """币种选择变化处理"""
        # 目前只支持一个币种，以后可以扩展
        self.load_and_plot_data()
    
    def on_interval_changed(self, index):
        """时间周期变化处理"""
        # 获取当前选择的时间周期值
        self.current_interval = self.interval_combo.currentData()
        self.load_and_plot_data()
    
    def on_range_changed(self, index):
        """时间范围变化处理"""
        self.load_and_plot_data()
    
    # 设置事件过滤器更新K线宽度显示
    def update_width_label(self):
        """更新K线宽度标签"""
        if hasattr(self, 'kline_canvas') and hasattr(self, 'width_label'):
            self.width_label.setText(f"K线宽度: {self.kline_canvas.bar_width:.1f}") 