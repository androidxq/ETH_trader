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
        ## 预先渲染K线图，提高绘制效率
        self.picture = QPicture()
        painter = QPainter(self.picture)
        painter.setPen(pg.mkPen('w'))
        
        # 使用固定宽度，确保K线紧挨着
        # 不再根据时间间隔计算宽度，而是使用传入的固定宽度
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
        self.full_data = None    # 完整数据
        self.current_data = None  # 当前显示的数据
        self.current_title = ""   # 当前标题
        
        # 视图范围控制
        self.max_display_count = 1000  # 最大显示K线数量
        self.load_threshold = 0.2      # 触发加载新数据的阈值
        
        # 创建布局
        self.setup_plots()
        
        # 启用抗锯齿
        self.setAntialiasing(True)
        
        # 添加视图范围变化监听
        self.price_plot.sigRangeChanged.connect(self.on_view_range_changed)
        
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
        
        # 移除错误的鼠标滚轮事件连接
        # 现在我们使用wheelEvent方法重写来处理滚轮事件
        
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

    def on_view_range_changed(self, view_box):
        """处理视图范围变化事件，动态加载和释放数据"""
        if self.full_data is None or len(self.full_data) == 0:
            return
            
        # 获取当前视图范围
        x_range = self.price_plot.viewRange()[0]
        current_width = x_range[1] - x_range[0]
        
        # 计算需要显示的数据范围
        start_idx = max(0, int(x_range[0] - current_width * self.load_threshold))
        end_idx = min(len(self.full_data) - 1, int(x_range[1] + current_width * self.load_threshold))
        
        # 如果当前显示的数据范围发生变化，重新加载数据
        if self.current_data is None or \
           start_idx < self.full_data.index(self.current_data[0]) or \
           end_idx > self.full_data.index(self.current_data[-1]):
            
            # 加载新的数据段
            display_count = min(self.max_display_count, end_idx - start_idx + 1)
            self.current_data = self.full_data[start_idx:start_idx + display_count]
            
            # 重新绘制K线图
            self.plot_kline(self.current_data, self.current_title)
            
            print(f"视图范围变化，加载新数据段: {start_idx} -> {start_idx + display_count}")
            print(f"当前显示 {len(self.current_data)} 条K线，总数据 {len(self.full_data)} 条")
    
    def wheelEvent(self, ev):
        """重写wheelEvent方法处理滚轮事件，控制视图缩放"""
        # 获取滚轮方向
        delta = ev.angleDelta().y()
        
        if self.current_data is None or len(self.current_data) == 0:
            return
            
        # 获取当前视图范围
        view_range = self.price_plot.viewRange()
        x_range = view_range[0]
        current_width = x_range[1] - x_range[0]
        
        # 获取鼠标位置作为缩放中心点
        center_x = (x_range[0] + x_range[1]) / 2
        
        # 计算新的视图宽度
        if delta > 0:  # 向上滚动，放大视图
            new_width = max(10, current_width * 0.8)
        else:  # 向下滚动，缩小视图
            new_width = min(self.max_display_count, current_width * 1.25)
        
        # 计算新的视图范围
        new_min = max(0, center_x - new_width / 2)
        new_max = min(len(self.current_data) - 1, center_x + new_width / 2)
        
        # 设置新的视图范围（使用平滑动画）
        self.setXRangeWithAnimation(new_min, new_max)
        
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
        """移动时间窗口，查看前后时间段的K线，支持动态加载数据
        
        Args:
            direction: 移动方向，1表示向前（查看更早数据），-1表示向后（查看更新数据）
        """
        if self.full_data is None or len(self.full_data) == 0:
            return
            
        # 更新时间偏移
        self.time_offset += direction
        
        # 获取当前视图范围
        view_range = self.price_plot.viewRange()
        x_range = view_range[0]
        
        # 计算移动的步数（显示宽度的一半）
        step = int((x_range[1] - x_range[0]) / 2)
        
        # 获取当前数据在完整数据中的位置
        current_start_idx = self.full_data.index(self.current_data[0]) if self.current_data else 0
        current_end_idx = current_start_idx + len(self.current_data) - 1
        
        # 计算新的数据范围
        new_start_idx = max(0, current_start_idx - step * direction)
        new_end_idx = min(len(self.full_data) - 1, current_end_idx - step * direction)
        
        # 确保至少显示一定数量的K线
        min_display_count = 30
        if new_end_idx - new_start_idx < min_display_count:
            if direction > 0:  # 向前移动
                new_end_idx = min(len(self.full_data) - 1, new_start_idx + min_display_count)
            else:  # 向后移动
                new_start_idx = max(0, new_end_idx - min_display_count)
        
        # 动态加载新的数据段
        self.current_data = self.full_data[new_start_idx:new_end_idx + 1]
        
        # 打印当前区域的时间范围
        if self.current_data:
            start_time = datetime.fromtimestamp(self.current_data[0][0])
            end_time = datetime.fromtimestamp(self.current_data[-1][0])
            print(f"加载新数据段: {start_time} -> {end_time}")
            print(f"数据范围: {new_start_idx} -> {new_end_idx}, 共{len(self.current_data)}条K线")
        
        # 更新Y轴范围（价格范围）
        try:
            data_np = np.array(self.current_data, dtype=float)
            max_price = np.max(data_np[:, 2])  # high
            min_price = np.min(data_np[:, 3])  # low
            
            # 计算价格范围，并添加边距
            price_range = max_price - min_price
            padding = price_range * 0.05
            
            # 设置价格轴范围
            self.price_plot.setYRange(min_price - padding, max_price + padding)
        except Exception as e:
            print(f"更新价格范围出错: {e}")
        
        # 重新绘制K线图
        self.plot_kline(self.current_data, self.current_title)
        
        # 设置新的显示范围（使用平滑动画）
        self.setXRangeWithAnimation(0, len(self.current_data))
        
        # 发出更新信号
        self.kline_updated.emit()
    
    def plot_kline(self, data, title="", max_display_count=1000):
        """绘制K线图，支持大数据量分段加载
        
        Args:
            data: 数据列表，每项为 [timestamp, open, high, low, close, volume]
            title: 图表标题
            max_display_count: 最大显示的K线数量，超过此数量将进行分段加载
        """
        print(f"\n===== 开始绘制K线图 =====")
        
        # 存储完整数据用于后续加载
        self.full_data = data
        self.current_title = title
        
        # 初始只加载最新的一部分数据
        display_count = min(max_display_count, len(data))
        self.current_data = data[-display_count:] if display_count > 0 else []
        
        print(f"加载最新的 {display_count} 条K线进行显示（总数据 {len(data)} 条）")
        
        # 清除现有图表内容
        self.price_plot.clear()
        self.volume_plot.clear()
        
        # 添加回十字光标
        self.price_plot.addItem(self.vLine, ignoreBounds=True)
        self.price_plot.addItem(self.hLine, ignoreBounds=True)
        
        # 如果数据为空则返回
        if not self.current_data or len(self.current_data) == 0:
            print("没有数据可供绘制")
            return
        
        # 打印当前显示的数据时间范围
        start_time = datetime.fromtimestamp(self.current_data[0][0])
        end_time = datetime.fromtimestamp(self.current_data[-1][0])
        print(f"当前显示K线范围：{start_time} 到 {end_time}")
        print(f"时间跨度：{(end_time - start_time).days} 天")
        
        # 设置视图范围监听
        self.price_plot.sigRangeChanged.connect(self.on_view_range_changed)
        
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
                except:
                    strings.append('')
            return strings
        
        # 修改价格图和成交量图的X轴刻度函数
        self.price_plot.getAxis('bottom').tickStrings = tickStrings
        self.volume_plot.getAxis('bottom').tickStrings = tickStrings
        
        # 默认显示最后N根K线（初始视图范围）
        display_count = min(30, len(indices))
        print(f"初始显示最后 {display_count} 根K线")
        
        # 计算当前显示区域内的价格范围
        visible_indices = indices[-display_count:]
        
        # 确保索引范围有效
        if len(visible_indices) > 0:
            visible_highs = highs[-display_count:]
            visible_lows = lows[-display_count:]
            
            # 获取可见区域内的最高价和最低价
            max_price = np.max(visible_highs)
            min_price = np.min(visible_lows)
            
            # 计算价格范围，并添加一定的上下边距（5%，减小边距让K线更高）
            price_range = max_price - min_price
            padding = price_range * 0.05
            
            # 设置价格轴范围，稍微放大一些使K线看起来更高
            self.price_plot.setYRange(min_price - padding, max_price + padding)
            print(f"优化价格范围: {min_price - padding:.2f} -> {max_price + padding:.2f}")
        
        # 计算初始显示范围 - 使用索引
        min_x = indices[-display_count] - 0.5  # 稍微留出左侧空间
        max_x = indices[-1] + 1.5  # 稍微留出右侧空间
        
        # 设置初始显示范围
        self.price_plot.setXRange(min_x, max_x)
        print(f"初始显示范围: 索引 {min_x:.1f} -> {max_x:.1f}")
        print(f"显示 {display_count} 条K线，总计加载 {len(indices)} 条K线")
        print("K线图绘制完成")

    def keyPressEvent(self, event):
        """处理键盘事件，支持快捷键控制K线图
        
        + / = 键增加显示的K线数量
        - 键减少显示的K线数量
        Left 键向前导航
        Right 键向后导航
        Home 键重置到默认视图
        """
        key = event.key()
        
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):  # + 或 = 键扩大显示范围
            self.zoom_out_view()
        elif key == Qt.Key.Key_Minus:  # - 键缩小显示范围
            self.zoom_in_view()
        elif key == Qt.Key.Key_Left:  # Left键向前导航
            self.move_time_window(1)
        elif key == Qt.Key.Key_Right:  # Right键向后导航
            self.move_time_window(-1)
        elif key == Qt.Key.Key_Home:  # Home键重置视图
            self.reset_view()
        else:
            super().keyPressEvent(event)
    
    def zoom_in_view(self):
        """缩小视图范围，显示更少的K线"""
        if self.current_data is None or len(self.current_data) == 0:
            return
            
        # 获取当前视图范围
        view_range = self.price_plot.viewRange()
        x_range = view_range[0]
        current_width = x_range[1] - x_range[0]
        
        # 缩小显示范围为当前的75%
        new_width = max(10, current_width * 0.75)  # 至少显示10根K线
        center = (x_range[0] + x_range[1]) / 2
        new_min = max(0, center - new_width / 2)
        new_max = min(len(self.current_data) - 1, center + new_width / 2)
        
        self.price_plot.setXRange(new_min, new_max)
        print(f"缩小显示范围: {new_min:.1f} -> {new_max:.1f}")
        
        # 发出更新信号
        self.kline_updated.emit()
    
    def zoom_out_view(self):
        """扩大视图范围，显示更多的K线"""
        if self.current_data is None or len(self.current_data) == 0:
            return
            
        # 获取当前视图范围
        view_range = self.price_plot.viewRange()
        x_range = view_range[0]
        current_width = x_range[1] - x_range[0]
        
        # 扩大显示范围为当前的150%
        new_width = min(len(self.current_data), current_width * 1.5)
        center = (x_range[0] + x_range[1]) / 2
        new_min = max(0, center - new_width / 2)
        new_max = min(len(self.current_data) - 1, center + new_width / 2)
        
        self.price_plot.setXRange(new_min, new_max)
        print(f"扩大显示范围: {new_min:.1f} -> {new_max:.1f}")
        
        # 发出更新信号
        self.kline_updated.emit()
    
    def reset_view(self):
        """重置视图到默认状态（显示最后N根K线）"""
        if self.current_data is None or len(self.current_data) == 0:
            return
            
        # 重置时间偏移
        self.time_offset = 0
        
        # 显示最后30根K线（与plot_kline方法一致）
        display_count = 30
        data_length = len(self.current_data)
        
        # 限制显示数量不超过数据总量
        display_count = min(display_count, data_length)
        
        # 计算显示范围 - 使用索引
        min_x = data_length - display_count - 0.5  # 稍微留出左侧空间
        max_x = data_length - 1 + 1.5  # 稍微留出右侧空间
        
        # 设置显示范围
        self.price_plot.setXRange(min_x, max_x)
        print(f"重置视图范围: {min_x:.1f} -> {max_x:.1f}")
        
        # 发出更新信号
        self.kline_updated.emit()

class KlineViewWidget(QWidget):
    """K线图显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loaded_data = {}  # 存储已加载的不同币种和时间周期的数据
        self.current_symbol = SYMBOL_CONFIG["main_symbol"]
        self.current_interval = "5m"  # 默认5分钟K线
        
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
        
        # 日期选择按钮
        self.date_button = QPushButton("跳转到日期")
        self.date_button.clicked.connect(self.on_jump_to_date)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.clicked.connect(self.load_and_plot_data)
        
        # 添加控件到网格布局
        control_layout.addWidget(symbol_label, 0, 0)
        control_layout.addWidget(self.symbol_combo, 0, 1)
        control_layout.addWidget(interval_label, 0, 2)
        control_layout.addWidget(self.interval_combo, 0, 3)
        control_layout.addWidget(self.date_button, 0, 4)
        control_layout.addWidget(self.refresh_button, 0, 5)
        
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
        self.width_label = QLabel("K线缩放: 正常")
        info_layout.addWidget(self.width_label)
        
        # 添加滚轮提示
        scroll_tip = QLabel("提示: 使用鼠标滚轮放大/缩小K线视图")
        scroll_tip.setStyleSheet("color: #2196F3;")
        info_layout.addWidget(scroll_tip)
        
        # 添加键盘快捷键提示
        keys_tip = QLabel("快捷键: +/- 缩放视图 | ← → 导航 | Home 重置")
        keys_tip.setStyleSheet("color: #2196F3;")
        info_layout.addWidget(keys_tip)
        
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
        """查看前一时间段（更早的数据）"""
        # 增加向前移动的距离
        if hasattr(self.kline_view, 'price_plot'):
            # 获取当前视图范围
            view_range = self.kline_view.price_plot.viewRange()
            x_range = view_range[0]
            view_width = x_range[1] - x_range[0]
            
            # 确定移动的距离（当前视图宽度的一半）
            move_step = int(view_width / 2)
            
            print(f"向前查看历史数据，移动步数: {move_step}")
            
            # 向前移动视图（更早的数据）
            self.kline_view.move_time_window(1)
    
    def on_next_time_clicked(self):
        """查看后一时间段（更新的数据）"""
        # 向后移动视图（更新的数据）
        self.kline_view.move_time_window(-1)
    
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
            print(f"使用缓存数据: {key}, 数据长度: {len(self.loaded_data[key])}")
            return self.loaded_data[key]
            
        # 加载数据
        data_pattern = f"data/kline/{symbol}_{interval}_*.csv"
        print(f"加载K线数据: {data_pattern}")
        
        # 列出匹配的文件
        import glob
        matching_files = glob.glob(data_pattern)
        print(f"找到 {len(matching_files)} 个匹配的数据文件:")
        for file in matching_files:
            print(f"  - {file}")
        
        # 加载数据文件
        df = load_data_files(data_pattern)
        
        # 如果数据为空，则返回空列表
        if df.empty:
            print(f"未找到数据: {data_pattern}")
            return []
        
        print(f"原始数据：{len(df)} 条记录")
        print(f"时间范围: {df['timestamp'].min()} - {df['timestamp'].max()}")
        
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
            
        # 确保数据是按时间排序的
        data_list.sort(key=lambda x: x[0])
        
        # 打印时间范围信息
        if data_list:
            start_time = datetime.fromtimestamp(data_list[0][0])
            end_time = datetime.fromtimestamp(data_list[-1][0])
            duration = end_time - start_time
            days = duration.days
            hours = duration.seconds // 3600
            print(f"成功加载 {len(data_list)} 条K线数据")
            print(f"时间范围：{start_time} - {end_time} (共 {days} 天 {hours} 小时)")
            
            # 打印时间分布统计
            if len(data_list) > 0:
                # 将数据按天分组
                from collections import defaultdict
                daily_counts = defaultdict(int)
                for item in data_list:
                    day = datetime.fromtimestamp(item[0]).strftime('%Y-%m-%d')
                    daily_counts[day] += 1
                
                print(f"数据按天分布统计 (前10天):")
                for i, (day, count) in enumerate(sorted(daily_counts.items())[:10]):
                    print(f"  - {day}: {count} 条K线")
                
                if len(daily_counts) > 10:
                    print(f"  ... 还有 {len(daily_counts) - 10} 天数据 ...")
        else:
            print("加载的数据列表为空")
            
        # 缓存数据
        self.loaded_data[key] = data_list
        return data_list
    
    def load_and_plot_data(self):
        """加载并绘制当前选择的K线数据"""
        # 获取当前选择
        symbol = self.current_symbol
        interval = self.current_interval
        
        # 加载数据
        print(f"\n===== 加载K线数据 {symbol} {interval} =====")
        data = self.load_data(symbol, interval)
        if not data:
            self.update_price_info(None)
            print("没有数据可供显示")
            return
            
        print(f"加载完成，数据长度: {len(data)} 条K线")
            
        # 设置图表标题
        title = f"{symbol} {INTERVAL_DISPLAY.get(interval, interval)} - 全部历史数据"
        
        # 始终使用全量数据进行绘制
        self.kline_view.plot_kline(data, title)
        
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
    
    def on_jump_to_date(self):
        """跳转到特定日期的K线"""
        # 创建日期选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择日期")
        layout = QVBoxLayout(dialog)
        
        # 创建日历控件
        calendar = QCalendarWidget()
        calendar.setGridVisible(True)
        
        # 设置日期范围
        data_for_range = self.kline_view.current_data
        
        if data_for_range:
            # 设置最早和最晚日期
            first_ts = data_for_range[0][0]
            last_ts = data_for_range[-1][0]
            
            first_date = QDateTime.fromSecsSinceEpoch(int(first_ts)).date()
            last_date = QDateTime.fromSecsSinceEpoch(int(last_ts)).date()
            
            print(f"设置日期选择器范围: {first_date.toString('yyyy-MM-dd')} 到 {last_date.toString('yyyy-MM-dd')}")
            
            calendar.setDateRange(first_date, last_date)
            calendar.setSelectedDate(last_date)  # 默认选择最新日期
        
        layout.addWidget(calendar)
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # 显示对话框
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_date = calendar.selectedDate()
            print(f"用户选择了日期: {selected_date.toString('yyyy-MM-dd')}")
            self.jump_to_date(selected_date)

    def jump_to_date(self, qt_date):
        """跳转到特定日期的K线
        
        Args:
            qt_date: PyQt6日期对象
        """
        # 首先确保有数据可供查看
        if not self.kline_view.current_data:
            print("没有可用数据进行跳转")
            return
        
        # 使用current_data
        target_data = self.kline_view.current_data
        
        # 将Qt日期转换为Python日期
        selected_date = qt_date.toPyDate()
        
        # 将日期转换为datetime，设定为当天开始时间
        target_dt = datetime.combine(selected_date, datetime.min.time())
        target_ts = target_dt.timestamp()
        
        print(f"跳转到日期: {selected_date} (时间戳: {target_ts})")
        
        # 获取时间戳列表
        timestamps = [item[0] for item in target_data]
        
        # 寻找最接近的时间戳
        closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_ts))
        closest_time = datetime.fromtimestamp(timestamps[closest_idx])
        
        print(f"找到最接近的K线索引: {closest_idx} / {len(timestamps)-1}")
        print(f"对应时间: {closest_time}")
        
        # 计算显示范围
        data_length = len(target_data)
        display_count = 30
        
        # 确保索引不超出范围
        start_idx = max(0, closest_idx - display_count // 4)  # 选择的日期左侧显示一些K线
        end_idx = min(data_length - 1, start_idx + display_count - 1)
        
        # 如果接近数据末尾，调整起始位置
        if end_idx >= data_length - 5:
            start_idx = max(0, data_length - display_count)
            end_idx = data_length - 1
        
        # 设置显示范围
        self.kline_view.price_plot.setXRange(start_idx - 0.5, end_idx + 1.5)
        
        # 更新时间偏移
        if closest_idx < data_length - 1:
            self.kline_view.time_offset = data_length - 1 - closest_idx
        else:
            self.kline_view.time_offset = 0
        
        print(f"设置显示范围: {start_idx} -> {end_idx}，时间偏移: {self.kline_view.time_offset}")
        
        # 更新信息显示
        self.update_position_label()
        
        # 刷新视图
        self.kline_view.kline_updated.emit()
    
    def update_width_label(self):
        """更新K线宽度标签"""
        if hasattr(self, 'kline_view') and hasattr(self, 'width_label'):
            # 获取当前视图范围
            view_range = self.kline_view.price_plot.viewRange()
            x_range = view_range[0]
            view_width = x_range[1] - x_range[0]
            
            # 根据显示K线数量更新缩放状态
            if view_width < 20:
                zoom_state = "放大"
            elif view_width > 50:
                zoom_state = "缩小"
            else:
                zoom_state = "正常"
                
            self.width_label.setText(f"K线缩放: {zoom_state} ({view_width:.0f}根)")