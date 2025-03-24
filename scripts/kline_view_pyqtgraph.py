"""
使用PyQtGraph实现的K线图显示模块

提供高性能、支持实时交互的K线图显示，支持多币种、多时间周期切换
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pyqtgraph as pg
from pyqtgraph import DateAxisItem

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入PyQt6
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QComboBox, QPushButton, QGroupBox, QSplitter, 
                            QFrame, QGridLayout, QSizePolicy, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QDateTime, QPointF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QPicture

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

# 自定义时间轴适配器，处理时间戳格式化
class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super(TimeAxisItem, self).__init__(*args, **kwargs)
        self.setLabel(text='时间', units=None)
        self.enableAutoSIPrefix(False)
        
    def tickStrings(self, values, scale, spacing):
        """处理时间轴刻度的字符串显示"""
        strings = []
        for value in values:
            try:
                dt = datetime.fromtimestamp(value)
                strings.append(dt.strftime('%m-%d %H:%M'))
            except:
                strings.append('')
        return strings

# K线图自定义Item，用于绘制单个K线
class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        super(CandlestickItem, self).__init__()
        self.data = data  # 数据格式应为: [timestamp, open, high, low, close, volume]
        self.picture = None
        self.generatePicture()
        
    def generatePicture(self):
        ## 预先渲染K线图，提高绘制效率
        self.picture = QPicture()
        painter = QPainter(self.picture)
        painter.setPen(pg.mkPen('w'))
        
        width = 0.8  # K线宽度 (基础值)
        for (t, open, high, low, close, volume) in self.data:
            # 确定K线颜色 (上涨为绿色，下跌为红色)
            if close > open:
                color = QColor('#26a69a')  # 绿色
            else:
                color = QColor('#ef5350')  # 红色
                
            # 设置画笔和画刷
            painter.setPen(pg.mkPen(color))
            
            # 绘制影线
            painter.drawLine(QPointF(t, low), QPointF(t, high))
            
            # 绘制实体
            rect = QRectF(
                QPointF(t - width/2, open),
                QPointF(t + width/2, close)
            )
            painter.setBrush(pg.mkBrush(color))
            painter.drawRect(rect)
            
        painter.end()
        
    def paint(self, painter, option, widget):
        painter.drawPicture(0, 0, self.picture)
        
    def boundingRect(self):
        return QRectF(self.picture.boundingRect())

class KlineGraphWidget(pg.GraphicsLayoutWidget):
    """基于PyQtGraph的高性能K线图组件"""
    
    # 自定义信号
    kline_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super(KlineGraphWidget, self).__init__(parent)
        
        # 设置白色背景
        self.setBackground('#1e1e1e')
        
        # K线显示相关设置
        self.bar_width = 0.8     # K线宽度
        self.min_bar_width = 0.3 # 最小K线宽度
        self.max_bar_width = 1.5 # 最大K线宽度
        
        # 时间轴导航参数
        self.time_offset = 0  # 时间偏移量，0表示最新数据
        
        # 数据存储
        self.current_data = None  # 当前显示的数据
        self.full_data = None     # 完整数据集
        self.current_title = ""   # 当前标题
        self.date_range = "1d"    # 当前显示的时间范围
        
        # 创建布局
        self.setup_plots()
        
        # 启用抗锯齿
        self.setAntialiasing(True)
        
    def setup_plots(self):
        """初始化图表布局"""
        # 清除现有布局
        self.clear()
        
        # 创建时间轴
        time_axis = TimeAxisItem(orientation='bottom')
        
        # 创建主K线图表
        self.price_plot = self.addPlot(row=0, col=0, axisItems={'bottom': time_axis})
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel('left', '价格')
        self.price_plot.setMinimumHeight(300)
        
        # 创建成交量图表
        self.volume_plot = self.addPlot(row=1, col=0, axisItems={'bottom': time_axis})
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setLabel('left', '成交量')
        self.volume_plot.setMaximumHeight(100)
        
        # 连接X轴范围
        self.price_plot.setXLink(self.volume_plot)
        
        # 添加十字光标
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.price_plot.addItem(self.vLine, ignoreBounds=True)
        self.price_plot.addItem(self.hLine, ignoreBounds=True)
        
        # 添加鼠标移动事件处理
        self.proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        
        # 添加鼠标滚轮事件处理
        self.price_plot.scene().sigMouseWheel.connect(self.mouseWheelEvent)
        
    def mouseMoved(self, evt):
        """鼠标移动事件处理，显示十字光标和数据提示"""
        pos = evt[0]  # 鼠标位置
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            self.vLine.setPos(mouse_point.x())
            self.hLine.setPos(mouse_point.y())
            
    def mouseWheelEvent(self, ev):
        """鼠标滚轮事件处理，缩放K线大小"""
        delta = ev.delta()
        print(f"捕获到PyQtGraph滚轮事件，delta={delta}")
        
        old_width = self.bar_width
        
        if delta > 0:  # 向上滚动，放大
            self.bar_width = min(self.bar_width + 0.1, self.max_bar_width)
            print(f"增大K线宽度为：{self.bar_width}")
        else:  # 向下滚动，缩小
            self.bar_width = max(self.bar_width - 0.1, self.min_bar_width)
            print(f"减小K线宽度为：{self.bar_width}")
            
        # 如果宽度真的变化了才重绘
        if abs(old_width - self.bar_width) > 0.01 and self.current_data is not None:
            print("重新绘制K线图")
            self.plot_kline(self.current_data, self.current_title)
            # 发出更新信号
            self.kline_updated.emit()
        
    def move_time_window(self, direction):
        """移动时间窗口，查看前后时间段的K线
        
        Args:
            direction: 移动方向，1表示向前（查看更早数据），-1表示向后（查看更新数据）
        """
        if self.full_data is None or len(self.full_data) == 0:
            return
            
        # 更新时间偏移
        self.time_offset += direction
        
        # 限制偏移范围，不超过数据集大小
        max_offset = len(self.full_data) // 10  # 假设一个合理的最大偏移
        self.time_offset = max(-max_offset, min(self.time_offset, 0))
        
        # 重新绘制数据
        if self.time_offset == 0:
            # 重置为默认范围（最新数据）
            self.plot_kline_with_range(self.full_data, self.current_title, self.date_range)
        else:
            # 计算新的时间窗口并绘制
            self._plot_with_offset()
        
        # 发出更新信号
        self.kline_updated.emit()
        
    def _plot_with_offset(self):
        """根据时间偏移绘制数据"""
        if self.full_data is None or len(self.full_data) == 0:
            return
            
        data_df = pd.DataFrame(self.full_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 根据当前时间范围计算窗口大小
        if self.date_range == "1d":
            window_size = timedelta(days=1)
            step_size = timedelta(hours=6)
        elif self.date_range == "3d":
            window_size = timedelta(days=3)
            step_size = timedelta(days=1)
        elif self.date_range == "1w":
            window_size = timedelta(days=7)
            step_size = timedelta(days=2)
        elif self.date_range == "2w":
            window_size = timedelta(days=14)
            step_size = timedelta(days=3)
        elif self.date_range == "1m":
            window_size = timedelta(days=30)
            step_size = timedelta(days=7)
        elif self.date_range == "3m":
            window_size = timedelta(days=90)
            step_size = timedelta(days=14)
        else:  # "all"
            self.plot_kline(self.full_data, f"{self.current_title}")
            return
            
        # 获取最新时间和偏移后的时间
        latest_time = pd.to_datetime(data_df['timestamp']).max()
        offset_time = latest_time + step_size * self.time_offset
        start_time = offset_time - window_size
        
        # 过滤数据
        mask = (pd.to_datetime(data_df['timestamp']) >= start_time) & \
               (pd.to_datetime(data_df['timestamp']) <= offset_time)
        filtered_data = data_df[mask].values.tolist()
        
        # 绘制过滤后的数据
        if filtered_data:
            title = f"{self.current_title} (偏移: {self.time_offset})"
            self.plot_kline(filtered_data, title)
    
    def plot_kline_with_range(self, data, title="", date_range="1d"):
        """根据时间范围绘制K线图
        
        Args:
            data: 数据列表，每项为 [timestamp, open, high, low, close, volume]
            title: 图表标题
            date_range: 范围字符串("1d", "3d", "1w", "2w", "1m", "3m", "all")
        """
        # 存储完整数据和范围设置
        self.full_data = data
        self.date_range = date_range
        self.time_offset = 0  # 重置时间偏移
        
        # 如果数据为空则返回
        if not data or len(data) == 0:
            return
            
        # 转换为DataFrame便于处理
        data_df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 根据时间范围筛选数据
        if date_range == "all":
            filtered_data = data
        else:
            # 获取当前最新数据的时间
            latest_time = pd.to_datetime(data_df['timestamp']).max()
            
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
            
            # 过滤数据
            mask = pd.to_datetime(data_df['timestamp']) >= start_time
            filtered_data = data_df[mask].values.tolist()
        
        # 绘制过滤后的数据
        self.plot_kline(filtered_data, title)
    
    def plot_kline(self, data, title=""):
        """绘制K线图
        
        Args:
            data: 数据列表，每项为 [timestamp, open, high, low, close, volume]
            title: 图表标题
        """
        # 存储当前数据用于后续操作
        self.current_data = data
        self.current_title = title
        
        # 清除现有图表内容
        self.price_plot.clear()
        self.volume_plot.clear()
        
        # 添加回十字光标
        self.price_plot.addItem(self.vLine, ignoreBounds=True)
        self.price_plot.addItem(self.hLine, ignoreBounds=True)
        
        # 如果数据为空则返回
        if not data or len(data) == 0:
            return
        
        # 设置标题
        self.price_plot.setTitle(title, color='#ffffff', size='12pt')
        
        # 准备处理数据
        data_np = np.array(data, dtype=float)
        timestamps = data_np[:, 0].astype(np.int64)
        opens = data_np[:, 1]
        highs = data_np[:, 2]
        lows = data_np[:, 3]
        closes = data_np[:, 4]
        volumes = data_np[:, 5]
        
        # 创建K线图项
        candlestick = CandlestickItem(data_np)
        self.price_plot.addItem(candlestick)
        
        # 创建成交量柱状图
        volume_brush = np.where(closes > opens, '#26a69a', '#ef5350')
        volume_brush = [pg.mkBrush(color) for color in volume_brush]
        volume_bar = pg.BarGraphItem(x=timestamps, height=volumes * 0.4, width=self.bar_width, brush=volume_brush)
        self.volume_plot.addItem(volume_bar)
        
        # 自动调整视图范围
        self.price_plot.autoRange()
        self.volume_plot.autoRange()


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
        self.kline_view = KlineGraphWidget(self)
        self.kline_view.kline_updated.connect(self.on_kline_updated)
        
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
        main_layout.addWidget(self.kline_view, 1)  # 1表示伸展因子
        main_layout.addWidget(info_group)
        
        # 初始加载数据
        self.load_and_plot_data()
    
    def on_kline_updated(self):
        """K线图更新后的处理函数"""
        self.update_width_label()
        self.update_position_label()
    
    def on_prev_time_clicked(self):
        """查看前一时间段"""
        self.kline_view.move_time_window(1)  # 向前移动
    
    def on_next_time_clicked(self):
        """查看后一时间段"""
        self.kline_view.move_time_window(-1)  # 向后移动
    
    def on_reset_time_clicked(self):
        """重置到最新数据"""
        if self.kline_view.time_offset != 0:
            self.kline_view.time_offset = 0
            self.load_and_plot_data()  # 重新加载当前数据
    
    def update_position_label(self):
        """更新位置标签"""
        offset = self.kline_view.time_offset
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
            list: 加载的数据，每项为 [timestamp, open, high, low, close, volume]
        """
        # 检查是否已经加载过
        key = f"{symbol}_{interval}"
        if key in self.loaded_data:
            return self.loaded_data[key]
            
        # 加载数据
        data_pattern = f"data/kline/{symbol}_{interval}_*.csv"
        df = load_data_files(data_pattern)
        
        # 如果数据为空，则返回空列表
        if df.empty:
            print(f"未找到数据: {data_pattern}")
            return []
        
        # 转换为列表格式，并将timestamp转换为Unix时间戳
        data_list = []
        for _, row in df.iterrows():
            ts = row['timestamp'].timestamp()  # 转换为Unix时间戳
            data_list.append([
                ts,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ])
            
        # 缓存数据
        self.loaded_data[key] = data_list
        return data_list
    
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
        if not data:
            self.update_price_info(None)
            return
            
        # 设置图表标题
        title = f"{symbol} {INTERVAL_DISPLAY.get(interval, interval)}"
        
        # 使用新的带范围绘制方法
        self.kline_view.plot_kline_with_range(data, title, date_range)
        
        # 如果有数据，更新最新价格信息
        latest_data = data[-1]  # 假设数据是按时间排序的
        self.update_price_info(latest_data)
        
        # 更新K线宽度显示和位置标签
        self.update_width_label()
        self.update_position_label()
    
    def update_price_info(self, latest_data):
        """更新价格信息面板
        
        Args:
            latest_data: 最新数据，格式为 [timestamp, open, high, low, close, volume]
        """
        if latest_data is None:
            # 清空价格信息
            for label_name in self.price_labels:
                self.price_labels[label_name].setText(f"{label_name}: -")
            return
            
        # 解包数据
        _, open_price, high, low, close, volume = latest_data
        
        # 更新各项数据
        self.price_labels["开盘价"].setText(f"开盘价: {open_price:.2f}")
        self.price_labels["最高价"].setText(f"最高价: {high:.2f}")
        self.price_labels["最低价"].setText(f"最低价: {low:.2f}")
        self.price_labels["收盘价"].setText(f"收盘价: {close:.2f}")
        self.price_labels["成交量"].setText(f"成交量: {volume:.2f}")
    
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
    
    def update_width_label(self):
        """更新K线宽度标签"""
        if hasattr(self, 'kline_view') and hasattr(self, 'width_label'):
            self.width_label.setText(f"K线宽度: {self.kline_view.bar_width:.1f}")