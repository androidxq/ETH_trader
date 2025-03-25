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
                            QFrame, QGridLayout, QSizePolicy, QApplication,
                            QDialog, QCalendarWidget, QDialogButtonBox)
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
    def __init__(self, data, width=0.8):
        super(CandlestickItem, self).__init__()
        self.data = data  # 数据格式应为: [timestamp, open, high, low, close, volume]
        self.picture = None
        self.width = width  # K线宽度
        self.generatePicture()
        
    def generatePicture(self):
        """预先渲染K线图，提高绘制效率"""
        self.picture = QPicture()
        painter = QPainter(self.picture)
        painter.setPen(pg.mkPen('w'))
        
        # 使用固定宽度，确保K线紧挨着
        width = self.width
            
        print(f"K线绘制宽度: {width}")
        
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
        self.bar_width = 0.9     # K线宽度（增大）
        self.min_bar_width = 0.3 # 最小K线宽度（降低到0.3）
        self.max_bar_width = 1.3 # 最大K线宽度（降低到1.3）
        
        # 时间轴导航参数
        self.time_offset = 0  # 时间偏移量，0表示最新数据
        
        # 数据存储
        self.current_data = None  # 当前显示的数据
        self.current_title = ""   # 当前标题
        
        # 创建布局
        self.setup_plots()
        
        # 启用抗锯齿
        self.setAntialiasing(True)
        
    def setup_plots(self):
        """初始化图表布局"""
        # 清除现有布局
        self.clear()
        
        # 为价格图表和成交量图表分别创建独立的时间轴
        price_time_axis = TimeAxisItem(orientation='bottom')
        volume_time_axis = TimeAxisItem(orientation='bottom')
        
        # 创建主K线图表
        self.price_plot = self.addPlot(row=0, col=0, axisItems={'bottom': price_time_axis})
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel('left', '价格')
        self.price_plot.setMinimumHeight(300)
        
        # 创建成交量图表
        self.volume_plot = self.addPlot(row=1, col=0, axisItems={'bottom': volume_time_axis})
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setLabel('left', '成交量')
        self.volume_plot.setMaximumHeight(100)
        
        # 连接X轴范围，使两个图表的X轴同步
        self.volume_plot.setXLink(self.price_plot)
        
        # 添加十字光标
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.price_plot.addItem(self.vLine, ignoreBounds=True)
        self.price_plot.addItem(self.hLine, ignoreBounds=True)
        
        # 添加鼠标移动事件处理
        self.proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        
    def setXRangeWithAnimation(self, min_x, max_x, duration=200):
        """平滑动画设置X轴范围
        
        Args:
            min_x: 新的最小X值
            max_x: 新的最大X值
            duration: 动画持续时间，毫秒
        """
        # 获取当前范围
        current_range = self.price_plot.viewRange()[0]
        
        # 使用平滑动画过渡到新的范围
        self.price_plot.setXRange(min_x, max_x, padding=0)

    def wheelEvent(self, ev):
        """重写wheelEvent方法处理滚轮事件，控制视图缩放而不是K线宽度"""
        # 获取滚轮方向
        delta = ev.angleDelta().y()
        print(f"捕获到鼠标滚轮事件，delta={delta}")
        
        if self.current_data is None or len(self.current_data) == 0:
            return
            
        # 获取当前视图范围
        view_range = self.price_plot.viewRange()
        x_range = view_range[0]
        current_width = x_range[1] - x_range[0]
        
        # 获取鼠标位置作为缩放中心点
        # 简化处理方式，直接使用当前视图的中心点作为缩放中心
        center_x = (x_range[0] + x_range[1]) / 2
        
        # 计算新的视图宽度
        if delta > 0:  # 向上滚动，放大视图（显示更少K线）
            new_width = max(10, current_width * 0.8)  # 缩小到当前的80%，但至少显示10根K线
            print(f"放大视图: {current_width:.1f} -> {new_width:.1f}")
        else:  # 向下滚动，缩小视图（显示更多K线）
            new_width = min(len(self.current_data), current_width * 1.25)  # 扩大到当前的125%
            print(f"缩小视图: {current_width:.1f} -> {new_width:.1f}")
        
        # 计算新的视图范围
        new_min = max(0, center_x - new_width / 2)
        new_max = min(len(self.current_data) - 1, center_x + new_width / 2)
        
        # 如果范围超出数据边界，调整保持宽度
        if new_min <= 0:
            new_max = min(len(self.current_data) - 1, new_width)
        if new_max >= len(self.current_data) - 1:
            new_min = max(0, len(self.current_data) - 1 - new_width)
        
        # 设置新的视图范围（使用平滑动画）
        self.setXRangeWithAnimation(new_min, new_max)
        print(f"调整视图范围: {new_min:.1f} -> {new_max:.1f}, 显示{new_max-new_min:.1f}条K线")
        
        # 发出更新信号
        self.kline_updated.emit()
        
        # 阻止事件传递给父类
        ev.accept()
        
    def mouseMoved(self, evt):
        """鼠标移动事件处理，显示十字光标和数据提示"""
        pos = evt[0]  # 鼠标位置
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            
            # 设置十字光标位置
            self.vLine.setPos(x)
            self.hLine.setPos(y)
            
            # 尝试获取当前位置的K线数据，显示更详细的信息
            if self.current_data is not None and len(self.current_data) > 0:
                try:
                    idx = int(round(x))
                    if 0 <= idx < len(self.current_data):
                        # 获取该位置的K线数据
                        _, open_price, high, low, close, volume = self.current_data[idx]
                        
                        # 转换为时间显示
                        time_stamp = self.index_to_ts.get(idx)
                        if time_stamp:
                            dt = datetime.fromtimestamp(time_stamp)
                            time_str = dt.strftime('%Y-%m-%d %H:%M')
                        else:
                            time_str = "未知时间"
                            
                        # 创建提示信息
                        tooltip = f"时间: {time_str}\n开盘: {open_price:.2f}\n最高: {high:.2f}\n最低: {low:.2f}\n收盘: {close:.2f}\n成交量: {volume:.2f}"
                        
                        # 这里可以添加一个浮动标签显示这些信息
                        # 由于pyqtgraph本身不直接支持跟随鼠标的工具提示，这里只在控制台打印
                        # print(tooltip)
                except Exception as e:
                    print(f"获取K线数据出错: {e}")
            
    def move_time_window(self, direction):
        """移动时间窗口，查看前后时间段的K线
        
        Args:
            direction: 移动方向，1表示向前（查看更早数据），-1表示向后（查看更新数据）
        """
        if self.current_data is None or len(self.current_data) == 0:
            return
            
        # 更新时间偏移
        self.time_offset += direction
        
        # 获取当前视图范围
        view_range = self.price_plot.viewRange()
        x_range = view_range[0]
        
        # 计算移动的步数（显示宽度的一半）
        step = int((x_range[1] - x_range[0]) / 2)
        
        # 限制范围不超出数据边界
        data_length = len(self.current_data)
        min_idx = max(0, int(x_range[0]) - step * direction)
        max_idx = min(data_length - 1, int(x_range[1]) - step * direction)
        
        print(f"移动时间窗口: direction={direction}, time_offset={self.time_offset}")
        print(f"当前视图范围: {x_range[0]:.1f} -> {x_range[1]:.1f}, step={step}")
        print(f"数据长度: {data_length}, 移动前索引范围: {int(x_range[0])} -> {int(x_range[1])}")
        print(f"新索引范围计算: min_idx = max(0, {int(x_range[0])} - {step} * {direction}) = {min_idx}")
        print(f"新索引范围计算: max_idx = min({data_length-1}, {int(x_range[1])} - {step} * {direction}) = {max_idx}")
        
        # 检查是否到达数据边界
        if direction > 0 and min_idx == 0:
            print("已到达数据最早边界")
            # 尝试扩大视图显示更多数据
            max_idx = min(data_length - 1, max_idx + 5)
        elif direction < 0 and max_idx == data_length - 1:
            print("已到达数据最新边界")
            # 尝试扩大视图显示更多数据
            min_idx = max(0, min_idx - 5)
        
        # 确保至少显示一定数量的K线
        if max_idx - min_idx < 10:
            if direction > 0:  # 向前移动
                max_idx = min(data_length - 1, min_idx + 10)
            else:  # 向后移动
                min_idx = max(0, max_idx - 10)
        
        # 打印当前区域的时间范围，帮助调试
        if 0 <= min_idx < data_length and 0 <= max_idx < data_length:
            start_time = datetime.fromtimestamp(self.current_data[min_idx][0])
            end_time = datetime.fromtimestamp(self.current_data[max_idx][0])
            print(f"移动到时间范围: {start_time} -> {end_time}, 索引范围: {min_idx} -> {max_idx}")
        
        # 当移动到边界附近时，更新Y轴范围（价格范围）
        try:
            # 获取当前可见区域的数据
            visible_indices = range(min_idx, max_idx + 1)
            data_np = np.array(self.current_data, dtype=float)
            
            if min_idx >= len(data_np) or max_idx >= len(data_np):
                print(f"警告：索引超出范围！min_idx={min_idx}, max_idx={max_idx}, data_length={len(data_np)}")
                # 修正索引范围
                min_idx = min(min_idx, len(data_np) - 1)
                max_idx = min(max_idx, len(data_np) - 1)
                visible_indices = range(min_idx, max_idx + 1)
            
            # 获取可见区域内的最高价和最低价
            visible_highs = data_np[visible_indices, 2]  # high
            visible_lows = data_np[visible_indices, 3]   # low
            
            max_price = np.max(visible_highs)
            min_price = np.min(visible_lows)
            
            # 计算价格范围，并添加边距
            price_range = max_price - min_price
            padding = price_range * 0.05
            
            # 设置价格轴范围
            self.price_plot.setYRange(min_price - padding, max_price + padding)
            print(f"更新价格范围: {min_price - padding:.2f} -> {max_price + padding:.2f}")
        except Exception as e:
            print(f"更新价格范围出错: {e}")
        
        # 设置新的显示范围（使用平滑动画）
        self.setXRangeWithAnimation(min_idx - 0.5, max_idx + 1.5)
        print(f"移动到索引: {min_idx} -> {max_idx}，显示{max_idx - min_idx + 1}条K线")
        
        # 发出更新信号
        self.kline_updated.emit()
    
    def plot_kline(self, data, title=""):
        """绘制K线图
        
        Args:
            data: 数据列表，每项为 [timestamp, open, high, low, close, volume]
            title: 图表标题
        """
        print(f"\n===== 开始绘制K线图 =====")
        print(f"数据长度：{len(data)} 条K线")
        
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
            print("没有数据可供绘制")
            return
        
        # 打印数据时间范围
        start_time = datetime.fromtimestamp(data[0][0])
        end_time = datetime.fromtimestamp(data[-1][0])
        print(f"K线时间范围：{start_time} 到 {end_time}")
        print(f"总共 {len(data)} 条K线，覆盖 {(end_time - start_time).days} 天")
        
        # 设置标题
        self.price_plot.setTitle(title, color='#ffffff', size='12pt')
        
        # 准备处理数据
        data_np = np.array(data, dtype=float)
        timestamps = data_np[:, 0].astype(np.int64)  # 实际时间戳，用于X轴标签
        opens = data_np[:, 1]
        highs = data_np[:, 2]
        lows = data_np[:, 3]
        closes = data_np[:, 4]
        volumes = data_np[:, 5]
        
        # 创建索引位置数组，用于绘制K线
        # 这样K线会紧挨着，不会有间隔
        indices = np.arange(len(timestamps))
        
        # 创建时间戳到索引的映射，用于数据显示和查询
        self.ts_to_index = dict(zip(timestamps, indices))
        self.index_to_ts = dict(zip(indices, timestamps))
        
        # 输出某些特定时间的索引，用于调试
        debug_times = [
            "2023-03-08 13:40:00",
            "2023-03-09 07:15:00",  # 用户报告这个时间点以前的数据无法显示
            "2023-03-10 05:40:00"
        ]
        print("时间索引映射（调试）:")
        for time_str in debug_times:
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                ts = dt.timestamp()
                closest_ts = min(timestamps, key=lambda x: abs(x - ts))
                closest_idx = self.ts_to_index[closest_ts]
                closest_time = datetime.fromtimestamp(closest_ts)
                print(f"  - {time_str} -> 最接近的索引: {closest_idx}, 对应时间: {closest_time}")
            except Exception as e:
                print(f"  - {time_str} -> 查找失败: {e}")
        
        # 生成新的K线数据 - 使用索引位置替代时间戳
        indexed_data = np.column_stack((indices, opens, highs, lows, closes, volumes))
        
        # 创建K线图项 - 使用索引位置绘制
        # 加载所有K线数据，而不仅仅是当前视图的数据（预加载）
        try:
            print(f"生成K线图元素，K线宽度：{self.bar_width}")
            candlestick = CandlestickItem(indexed_data, self.bar_width)
            self.price_plot.addItem(candlestick)
            print("K线图元素添加成功")
        except Exception as e:
            print(f"生成K线图元素失败: {e}")
        
        # 创建成交量柱状图 - 分别绘制上涨和下跌的成交量
        up_idx = closes > opens
        down_idx = ~up_idx
        
        # 上涨成交量 (绿色)
        if np.any(up_idx):
            try:
                up_volume_bar = pg.BarGraphItem(
                    x=indices[up_idx], 
                    height=volumes[up_idx] * 0.4, 
                    width=self.bar_width, 
                    brush='#26a69a'  # 绿色
                )
                self.volume_plot.addItem(up_volume_bar)
                print(f"添加上涨成交量柱状图，共{np.sum(up_idx)}条")
            except Exception as e:
                print(f"添加上涨成交量失败: {e}")
        
        # 下跌成交量 (红色)
        if np.any(down_idx):
            try:
                down_volume_bar = pg.BarGraphItem(
                    x=indices[down_idx], 
                    height=volumes[down_idx] * 0.4, 
                    width=self.bar_width, 
                    brush='#ef5350'  # 红色
                )
                self.volume_plot.addItem(down_volume_bar)
                print(f"添加下跌成交量柱状图，共{np.sum(down_idx)}条")
            except Exception as e:
                print(f"添加下跌成交量失败: {e}")
        
        # 创建自定义X轴刻度
        def tickStrings(values, scale, spacing):
            """将索引位置转换回时间显示"""
            strings = []
            for value in values:
                try:
                    # 找到最接近的索引
                    idx = int(round(value))
                    if idx >= 0 and idx < len(timestamps):
                        # 转换回时间显示
                        dt = datetime.fromtimestamp(timestamps[idx])
                        strings.append(dt.strftime('%m-%d %H:%M'))
                    else:
                        strings.append('')
                except Exception as e:
                    print(f"时间转换错误: {e}")
                    strings.append('')
            return strings
        
        # 设置X轴刻度函数
        self.price_plot.getAxis('bottom').tickStrings = tickStrings
        self.volume_plot.getAxis('bottom').tickStrings = tickStrings
        
        # 设置初始显示范围
        display_count = min(30, len(indices))  # 默认显示30根K线
        self.price_plot.setXRange(len(indices) - display_count, len(indices) - 1)
        self.volume_plot.setXRange(len(indices) - display_count, len(indices) - 1)
        
        # 设置Y轴范围
        price_range = highs.max() - lows.min()
        price_padding = price_range * 0.05  # 5%的padding
        self.price_plot.setYRange(
            lows.min() - price_padding,
            highs.max() + price_padding
        )
        
        volume_range = volumes.max()
        volume_padding = volume_range * 0.05
        self.volume_plot.setYRange(0, volumes.max() + volume_padding)
        
        print("K线图绘制完成") 