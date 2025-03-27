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
                            QDialog, QCalendarWidget, QDialogButtonBox, QMessageBox)
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

def find_closest_index(data, target_ts):
    """找到最接近目标时间戳的索引
    
    Args:
        data: 数据列表，每项为 [timestamp, open, high, low, close, volume]
        target_ts: 目标时间戳
        
    Returns:
        int: 最接近的索引
    """
    if not data:
        return 0
    
    low, high = 0, len(data) - 1
    closest_idx = 0
    min_diff = float('inf')
    
    while low <= high:
        mid = (low + high) // 2
        mid_ts = data[mid][0]
        
        # 更新最接近的索引
        current_diff = abs(mid_ts - target_ts)
        if current_diff < min_diff:
            min_diff = current_diff
            closest_idx = mid
        
        # 二分查找逻辑
        if mid_ts < target_ts:
            low = mid + 1
        else:
            high = mid - 1
    
    return closest_idx

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
                # 根据间隔确定显示格式
                if spacing > 3600 * 24 * 90:  # 超过90天的间隔
                    # 显示年月
                    strings.append(dt.strftime('%Y-%m'))
                elif spacing > 3600 * 24 * 2:  # 超过2天的间隔
                    # 显示月日
                    strings.append(dt.strftime('%m-%d'))
                elif spacing > 3600 * 2:  # 超过2小时的间隔
                    # 显示日期和小时
                    strings.append(dt.strftime('%d %H:%M'))
                elif spacing > 60 * 30:  # 超过30分钟的间隔
                    # 只显示小时和分钟
                    strings.append(dt.strftime('%H:%M'))
                else:
                    # 显示分钟和秒
                    strings.append(dt.strftime('%H:%M:%S'))
            except Exception as e:
                print(f"时间轴刻度格式化错误: {e}, value={value}")
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
        
        # 用自定义Y轴替换默认Y轴
        left_axis = self.price_plot.getAxis('left')
        self.price_plot.layout.removeItem(left_axis)
        
        # 创建自定义价格Y轴
        price_y_axis = PriceYAxisItem(orientation='left', view_box=self.price_plot.getViewBox())
        self.price_plot.setAxisItems({'left': price_y_axis})
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

    def on_view_range_changed(self):
        """处理视图范围变化事件"""
        try:
            # 获取当前视图索引范围
            view_min_x, view_max_x = self.price_plot.getViewBox().viewRange()[0]
            view_width = view_max_x - view_min_x
            view_center = (view_min_x + view_max_x) / 2
            
            print(f"当前视图索引范围: {view_min_x} -> {view_max_x}, 宽度: {view_width}")
            
            # 如果当前没有数据，则无法处理
            if not self.current_data or len(self.current_data) == 0:
                print("当前没有数据，无法处理视图范围变化")
                return
                
            # 处理负索引情况 - 负索引表示更早的数据
            min_x = max(0, int(view_min_x))
            max_x = min(len(self.current_data) - 1, int(view_max_x))
            
            # 计算当前可视的时间范围
            visible_start_ts = self.current_data[min_x][0] if min_x < len(self.current_data) else self.current_data[-1][0]
            visible_end_ts = self.current_data[max_x][0] if max_x >= 0 else self.current_data[0][0]
            
            # 处理负索引对应的时间戳估算
            if view_min_x < 0:
                # 如果有负索引，估算对应的时间戳
                # 使用当前数据的第一条和第二条记录计算时间间隔
                if len(self.current_data) >= 2:
                    time_interval = self.current_data[1][0] - self.current_data[0][0]
                    # 估算负索引对应的时间戳
                    visible_start_ts = self.current_data[0][0] + view_min_x * time_interval
            
            visible_start_time = datetime.fromtimestamp(visible_start_ts)
            visible_end_time = datetime.fromtimestamp(visible_end_ts)
            print(f"当前可视时间范围: {visible_start_time.strftime('%Y-%m-%d %H:%M')} -> {visible_end_time.strftime('%Y-%m-%d %H:%M')}")
            
            # 更新界面显示的时间范围
            if self.parent:
                self.parent().update_date_range_label(visible_start_time, visible_end_time)
                
            # 计算视图中心对应的时间戳
            if min_x <= max_x and min_x >= 0 and max_x < len(self.current_data):
                view_center_idx = int(view_center)
                if view_center_idx >= 0 and view_center_idx < len(self.current_data):
                    view_center_ts = self.current_data[view_center_idx][0]
                else:
                    # 如果视图中心超出范围，使用估算
                    if view_center < 0:
                        # 负索引，估算更早的时间
                        time_interval = self.current_data[1][0] - self.current_data[0][0] if len(self.current_data) >= 2 else 300
                        view_center_ts = self.current_data[0][0] + view_center * time_interval
                    else:
                        # 超出右侧，使用最后一条记录的时间
                        view_center_ts = self.current_data[-1][0]
            else:
                # 如果没有可见数据，则使用当前数据的中间点
                middle_idx = len(self.current_data) // 2
                view_center_ts = self.current_data[middle_idx][0]
                
            # 如果没有全量数据或者全量数据为空，则无法继续处理
            if not hasattr(self, 'full_data') or not self.full_data or len(self.full_data) == 0:
                print("没有全量数据，无法加载更多数据")
                return
                
            # 打印全量数据的范围
            print(f"总索引范围: 0 -> {len(self.full_data) - 1}, 总数据量: {len(self.full_data)}")
            
            # 根据可视的时间范围，计算目标数据范围在全量数据中的索引
            target_start_idx = find_closest_index(self.full_data, visible_start_ts)
            target_end_idx = find_closest_index(self.full_data, visible_end_ts)
            
            # 添加一个缓冲区，确保负索引区域也能正确加载
            buffer_size = int(view_width)  # 使用视图宽度作为缓冲区大小
            target_start_idx = max(0, target_start_idx - buffer_size)
            target_end_idx = min(len(self.full_data) - 1, target_end_idx + buffer_size)
            
            print(f"目标索引范围: {target_start_idx} -> {target_end_idx}")
            
            # 获取当前数据在全量数据中的索引范围
            current_start_ts = self.current_data[0][0]
            current_end_ts = self.current_data[-1][0]
            current_start_idx = find_closest_index(self.full_data, current_start_ts)
            current_end_idx = find_closest_index(self.full_data, current_end_ts)
            print(f"当前数据索引范围: {current_start_idx} -> {current_end_idx}")
            
            # 判断是否需要加载新数据
            need_to_load = (target_start_idx < current_start_idx - 10 or  
                           target_end_idx > current_end_idx + 10 or
                           target_end_idx - target_start_idx > 2 * (current_end_idx - current_start_idx))
                           
            # 额外检查：如果视图有负索引且数据不足，强制加载
            if view_min_x < 0 and min_x == 0:
                need_to_load = True
                print("视图有负索引且数据不足，强制加载更多数据")
            
            if need_to_load:
                print(f"需要加载新数据，范围：{target_start_idx} -> {target_end_idx}")
                try:
                    # 加载新数据
                    new_data = self.full_data[target_start_idx:target_end_idx+1]
                    
                    if len(new_data) > 0:
                        # 更新当前数据
                        prev_data_len = len(self.current_data)
                        self.current_data = new_data
                        
                        # 如果有当前数据，打印时间范围
                        if len(self.current_data) > 0:
                            new_start_time = datetime.fromtimestamp(self.current_data[0][0])
                            new_end_time = datetime.fromtimestamp(self.current_data[-1][0])
                            print(f"当前加载数据范围: {new_start_time.strftime('%Y-%m-%d %H:%M')} -> {new_end_time.strftime('%Y-%m-%d %H:%M')}")
                        
                        # 绘制当前数据，不重新绘制全部K线图
                        self.plot_current_data()
                        
                        # 找到新数据中与原视图中心时间戳最接近的索引
                        new_center_idx = find_closest_index(self.current_data, view_center_ts)
                        
                        # 设置新的视图范围，保持之前的视图宽度
                        self.price_plot.setXRange(
                            max(0, new_center_idx - view_width/2),
                            min(len(self.current_data) - 1, new_center_idx + view_width/2),
                            padding=0
                        )
                except Exception as e:
                    print(f"加载新数据时出错: {e}")
            else:
                print("无需加载新数据")
                
        except Exception as e:
            print(f"视图范围变化处理出错: {e}")
            import traceback
            traceback.print_exc()
    
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
            if self.current_data is not None:
                try:
                    idx = int(round(x))
                    # 检查鼠标是否在加载的数据范围内
                    if 0 <= idx < len(self.current_data):
                        # 获取该位置的K线数据
                        _, open_price, high, low, close, volume = self.current_data[idx]
                        
                        # 转换为时间显示
                        time_stamp = self.index_to_ts.get(idx)
                        if time_stamp:
                            dt = datetime.fromtimestamp(time_stamp)
                            time_str = dt.strftime('%Y-%m-%d %H:%M')
                            
                            # 打印当前鼠标位置的K线信息
                            print(f"\r当前K线: 索引={idx}, 时间={time_str}, 价格={close:.2f}", end="")
                        else:
                            time_str = "未知时间"
                            
                        # 创建提示信息
                        tooltip = f"时间: {time_str}\n开盘: {open_price:.2f}\n最高: {high:.2f}\n最低: {low:.2f}\n收盘: {close:.2f}\n成交量: {volume:.2f}"
                    else:
                        # 鼠标在已加载数据范围外，但仍在图表区域
                        # 检查是否有完整数据可供参考
                        if self.full_data and len(self.full_data) > 0:
                            # 获取全量数据的时间范围
                            all_start_time = datetime.fromtimestamp(self.full_data[0][0])
                            all_end_time = datetime.fromtimestamp(self.full_data[-1][0])
                            
                            # 尝试估算鼠标位置的时间
                            if idx < 0:  # 左侧区域
                                # 在左侧超出范围，尝试估算日期
                                if len(self.current_data) > 1:
                                    # 使用第一个和第二个数据点计算时间间隔
                                    first_ts = self.current_data[0][0]
                                    second_ts = self.current_data[1][0]
                                    interval = second_ts - first_ts
                                    
                                    # 估算鼠标位置的时间
                                    estimated_ts = first_ts + idx * interval
                                    estimated_time = datetime.fromtimestamp(estimated_ts)
                                    print(f"\r估算日期: {estimated_time.strftime('%Y-%m-%d %H:%M')} (负向偏移: {idx}个单位)", end="")
                                
                                print(f"\r需要加载更早的数据 (当前索引: {idx}, 全量数据范围: {all_start_time.strftime('%Y-%m-%d')} 至 {all_end_time.strftime('%Y-%m-%d')})", end="")
                                
                                # 检查是否已经到达全量数据的最左端
                                if self.current_data[0][0] <= self.full_data[0][0] + 10:  # 允许有10秒误差
                                    print(f"\r*** 已经是最早数据 ({all_start_time.strftime('%Y-%m-%d %H:%M')}) ***", end="")
                                else:
                                    print(f"\r*** 建议点击「上一时间段」按钮加载更早数据或使用「跳转到日期」 ***", end="")
                            elif idx >= len(self.current_data):  # 右侧区域
                                # 在右侧超出范围，尝试估算日期
                                if len(self.current_data) > 1:
                                    # 使用最后两个数据点计算时间间隔
                                    last_idx = len(self.current_data) - 1
                                    last_ts = self.current_data[last_idx][0]
                                    prev_ts = self.current_data[last_idx-1][0]
                                    interval = last_ts - prev_ts
                                    
                                    # 估算鼠标位置的时间
                                    offset = idx - last_idx
                                    estimated_ts = last_ts + offset * interval
                                    estimated_time = datetime.fromtimestamp(estimated_ts)
                                    print(f"\r估算日期: {estimated_time.strftime('%Y-%m-%d %H:%M')} (正向偏移: {offset}个单位)", end="")
                                
                                print(f"\r需要加载更新的数据 (当前索引: {idx}, 全量数据范围: {all_start_time.strftime('%Y-%m-%d')} 至 {all_end_time.strftime('%Y-%m-%d')})", end="")
                                
                                # 如果是在最新数据右侧，提示已经是最新
                                if self.current_data[-1][0] >= self.full_data[-1][0] - 10:  # 允许有10秒误差
                                    print(f"\r*** 已经是最新数据 ({all_end_time.strftime('%Y-%m-%d %H:%M')}) ***", end="")
                                else:
                                    print(f"\r*** 建议点击「下一时间段」按钮加载更新数据 ***", end="")
                            
                            # 提示当前数据的时间范围
                            if self.current_data and len(self.current_data) > 0:
                                current_start_ts = self.current_data[0][0]
                                current_end_ts = self.current_data[-1][0]
                                start_time = datetime.fromtimestamp(current_start_ts)
                                end_time = datetime.fromtimestamp(current_end_ts)
                                print(f"\r当前加载数据范围: {start_time.strftime('%Y-%m-%d %H:%M')} -> {end_time.strftime('%Y-%m-%d %H:%M')}", end="")
                        else:
                            print(f"\r无数据可显示 (鼠标位置索引: {idx})", end="")
                except Exception as e:
                    print(f"\r获取K线数据出错: {e}", end="")
            
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
        
        # 打印导航前的位置
        print(f"\n--------- 移动时间窗口 {direction}，当前偏移：{self.time_offset} ---------")
        if self.current_data:
            current_start_time = datetime.fromtimestamp(self.current_data[0][0])
            current_end_time = datetime.fromtimestamp(self.current_data[-1][0])
            print(f"导航前时间范围: {current_start_time.strftime('%Y-%m-%d %H:%M')} 至 {current_end_time.strftime('%Y-%m-%d %H:%M')}")
        
        # 计算移动的步数（显示宽度的一半）
        step = int((x_range[1] - x_range[0]) / 2)
        
        # 获取当前数据在完整数据中的位置
        try:
            current_start_idx = self.full_data.index(self.current_data[0]) if self.current_data else 0
            current_end_idx = current_start_idx + len(self.current_data) - 1
            
            # 计算新的数据范围
            new_start_idx = max(0, current_start_idx - step * direction)
            new_end_idx = min(len(self.full_data) - 1, current_end_idx - step * direction)
            
            print(f"计算新范围: 从 [{current_start_idx}:{current_end_idx}] 移动到 [{new_start_idx}:{new_end_idx}]")
            
            # 确保至少显示一定数量的K线
            min_display_count = 30
            if new_end_idx - new_start_idx < min_display_count:
                if direction > 0:  # 向前移动
                    new_end_idx = min(len(self.full_data) - 1, new_start_idx + min_display_count)
                else:  # 向后移动
                    new_start_idx = max(0, new_end_idx - min_display_count)
                
                print(f"调整后的范围: [{new_start_idx}:{new_end_idx}]")
            
            # 检查是否已经到达数据边界
            at_boundary = False
            if direction > 0 and new_start_idx == 0:  # 已经到最早数据
                print("已到达当前加载数据的最早边界")
                at_boundary = True
            elif direction < 0 and new_end_idx >= len(self.full_data) - 1:  # 已经到最新数据
                print("已到达当前加载数据的最新边界")
                at_boundary = True
                
            # 如果到达边界，尝试加载更多数据
            if at_boundary and self._maybe_load_more_data(direction):
                # 成功加载了更多数据，重新规划范围
                return  # 函数已经在_maybe_load_more_data中处理了数据加载和显示
                
            # 动态加载新的数据段
            self.current_data = self.full_data[new_start_idx:new_end_idx + 1]
            
            # 打印当前区域的时间范围
            if self.current_data:
                start_time = datetime.fromtimestamp(self.current_data[0][0])
                end_time = datetime.fromtimestamp(self.current_data[-1][0])
                print(f"加载新数据段: {start_time.strftime('%Y-%m-%d %H:%M')} -> {end_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"数据范围: {new_start_idx} -> {new_end_idx}, 共{len(self.current_data)}条K线")
                
                # 打印一些采样数据点，确认加载正确
                sample_indices = [0, len(self.current_data)//4, len(self.current_data)//2, 3*len(self.current_data)//4, len(self.current_data)-1]
                print("采样数据点:")
                for idx in sample_indices:
                    if 0 <= idx < len(self.current_data):
                        ts = self.current_data[idx][0]
                        dt = datetime.fromtimestamp(ts)
                        print(f"  [索引 {idx}]: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
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
            self.plot_current_data()
            
            # 设置新的显示范围（使用平滑动画）
            self.setXRangeWithAnimation(0, len(self.current_data))
            
            # 发出更新信号
            self.kline_updated.emit()
            
        except ValueError as e:
            print(f"导航出错 - 无法定位当前数据位置: {e}")
            # 数据不在完整数据集中，可能需要重新加载
            if self._maybe_load_more_data(direction):
                # 成功加载了更多数据
                pass
            else:
                print("无法继续导航，当前数据不在完整数据集中")
    
    def _maybe_load_more_data(self, direction):
        """尝试加载更多数据
        
        Args:
            direction: 1表示加载更早的数据，-1表示加载更新的数据
            
        Returns:
            bool: 是否成功加载了更多数据
        """
        # 此方法由KlineGraphWidget类调用，需要访问父组件的方法
        if not hasattr(self, 'parent') or not self.parent():
            return False
            
        parent = self.parent()
        if not hasattr(parent, 'load_data') or not hasattr(parent, 'current_symbol') or not hasattr(parent, 'current_interval'):
            return False
            
        # 确定要加载的目标日期
        target_date = None
        if self.current_data and len(self.current_data) > 0:
            if direction > 0:  # 加载更早的数据
                # 获取当前最早数据的日期
                earliest_ts = self.current_data[0][0]
                earliest_date = datetime.fromtimestamp(earliest_ts).date()
                # 往前推7天，确保有重叠
                target_date = earliest_date - timedelta(days=7)
                print(f"尝试加载更早的数据，目标日期: {target_date}")
            else:  # 加载更新的数据
                # 获取当前最新数据的日期
                latest_ts = self.current_data[-1][0]
                latest_date = datetime.fromtimestamp(latest_ts).date()
                # 往后推7天，确保有重叠
                target_date = latest_date + timedelta(days=7)
                print(f"尝试加载更新的数据，目标日期: {target_date}")
        
        if target_date:
            # 尝试加载目标日期的数据
            new_data = parent.load_data(parent.current_symbol, parent.current_interval, target_date)
            
            if new_data and len(new_data) > 0:
                # 检查是否有重叠，如果没有重叠，可能是有数据缺失
                has_overlap = False
                
                if self.current_data and len(self.current_data) > 0:
                    current_timestamps = set(item[0] for item in self.current_data)
                    new_timestamps = set(item[0] for item in new_data)
                    common_timestamps = current_timestamps.intersection(new_timestamps)
                    
                    if common_timestamps:
                        has_overlap = True
                        print(f"新数据与当前数据有 {len(common_timestamps)} 条重叠记录")
                    else:
                        print("警告：新数据与当前数据没有重叠部分，可能存在数据缺失")
                
                # 更新完整数据集
                full_data_set = set()
                if self.full_data:
                    full_data_set = set(item[0] for item in self.full_data)
                
                # 添加新数据
                added_count = 0
                for item in new_data:
                    if item[0] not in full_data_set:
                        self.full_data.append(item)
                        added_count += 1
                
                # 重新排序完整数据
                if self.full_data:
                    self.full_data.sort(key=lambda x: x[0])
                
                print(f"添加了 {added_count} 条新数据")
                
                # 设置要显示的时间段
                if direction > 0:  # 显示更早的数据
                    # 获取最早数据的前30条
                    early_bound = 30
                    display_start = max(0, early_bound - 30)
                    display_end = min(len(self.full_data) - 1, early_bound + 30)
                    self.current_data = self.full_data[display_start:display_end + 1]
                else:  # 显示更新的数据
                    # 获取最新数据的最后30条
                    display_count = 30
                    display_start = max(0, len(self.full_data) - display_count)
                    display_end = len(self.full_data) - 1
                    self.current_data = self.full_data[display_start:display_end + 1]
                
                # 重新绘制图表
                title = parent.current_title if hasattr(parent, 'current_title') else ""
                self.plot_kline(self.full_data, title)
                
                # 设置视图范围
                if direction > 0:  # 显示更早的数据
                    self.setXRangeWithAnimation(0, 60)  # 显示前60条
                else:  # 显示更新的数据
                    self.setXRangeWithAnimation(len(self.current_data) - 60, len(self.current_data))  # 显示后60条
                
                # 发出更新信号
                self.kline_updated.emit()
                return True
        
        return False
    
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
        print(f"时间跨度：{(end_time - start_time).days} 天 {(end_time - start_time).seconds // 3600} 小时")
        
        # 设置视图范围监听
        self.price_plot.sigRangeChanged.connect(self.on_view_range_changed)
        
        # 设置标题 - 添加全量数据的时间范围信息
        # 使用父组件中存储的完整时间范围信息
        if hasattr(self.parent(), 'full_time_range'):
            all_start_time = self.parent().full_time_range['start_time']
            all_end_time = self.parent().full_time_range['end_time']
            duration = all_end_time - all_start_time
            
            # 构建更详细的标题
            title_parts = [
                title,
                f"[{all_start_time.strftime('%Y-%m-%d')} 至 {all_end_time.strftime('%Y-%m-%d')}]",
                f"共 {len(self.full_data)} 条K线",
                f"跨度 {duration.days} 天 {duration.seconds // 3600} 小时"
            ]
            title_with_range = " | ".join(title_parts)
            self.price_plot.setTitle(title_with_range, color='#ffffff', size='12pt')
        else:
            # 如果没有完整时间范围信息，使用当前数据的时间范围
            all_start_time = start_time
            all_end_time = end_time
            duration = all_end_time - all_start_time
            title_with_range = f"{title} [{all_start_time.strftime('%Y-%m-%d')} 至 {all_end_time.strftime('%Y-%m-%d')}]"
            self.price_plot.setTitle(title_with_range, color='#ffffff', size='12pt')
        
        # 准备处理数据
        data_np = np.array(self.current_data, dtype=float)
        timestamps = data_np[:, 0].astype(np.int64)  # 实际时间戳，用于X轴标签
        opens = data_np[:, 1]
        highs = data_np[:, 2]
        lows = data_np[:, 3]
        closes = data_np[:, 4]
        volumes = data_np[:, 5]
        
        # 创建索引位置数组，用于绘制K线
        indices = np.arange(len(timestamps))
        
        # 创建时间戳到索引的映射，用于数据显示和查询
        self.ts_to_index = dict(zip(timestamps, indices))
        self.index_to_ts = dict(zip(indices, timestamps))
        
        # 生成新的K线数据 - 使用索引位置替代时间戳
        indexed_data = np.column_stack((indices, opens, highs, lows, closes, volumes))
        
        # 使用更高效的方式创建K线图元素
        try:
            print(f"生成K线图元素，K线宽度：{self.bar_width}")
            # 优化K线渲染 - 仅当数据量大于1000条时使用更高效的绘制方式
            if len(indexed_data) > 1000:
                # 通过抽样或合并减少绘制元素数量
                sampling_rate = max(1, len(indexed_data) // 1000)
                if sampling_rate > 1:
                    print(f"数据量过大，执行1:{sampling_rate}抽样以提高性能")
                    # 进行抽样
                    sampled_data = indexed_data[::sampling_rate]
                    candlestick = CandlestickItem(sampled_data, self.bar_width * sampling_rate)
                else:
                    candlestick = CandlestickItem(indexed_data, self.bar_width)
            else:
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
                # 优化成交量图形渲染 - 同样对大数据量进行抽样
                if len(indexed_data) > 1000 and sampling_rate > 1:
                    up_indices = indices[up_idx][::sampling_rate]
                    up_volumes = volumes[up_idx][::sampling_rate] * 0.4
                    up_width = self.bar_width * sampling_rate
                else:
                    up_indices = indices[up_idx]
                    up_volumes = volumes[up_idx] * 0.4
                    up_width = self.bar_width
                    
                up_volume_bar = pg.BarGraphItem(
                    x=up_indices, 
                    height=up_volumes, 
                    width=up_width, 
                    brush='#26a69a'  # 绿色
                )
                self.volume_plot.addItem(up_volume_bar)
                print(f"添加上涨成交量柱状图，共{np.sum(up_idx)}条")
            except Exception as e:
                print(f"添加上涨成交量失败: {e}")
        
        # 下跌成交量 (红色)
        if np.any(down_idx):
            try:
                # 优化成交量图形渲染
                if len(indexed_data) > 1000 and sampling_rate > 1:
                    down_indices = indices[down_idx][::sampling_rate]
                    down_volumes = volumes[down_idx][::sampling_rate] * 0.4
                    down_width = self.bar_width * sampling_rate
                else:
                    down_indices = indices[down_idx]
                    down_volumes = volumes[down_idx] * 0.4
                    down_width = self.bar_width
                    
                down_volume_bar = pg.BarGraphItem(
                    x=down_indices, 
                    height=down_volumes, 
                    width=down_width, 
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
            prev_year = None  # 跟踪前一个年份
            
            for value in values:
                try:
                    # 找到最接近的索引
                    idx = int(round(value))
                    if idx >= 0 and idx < len(timestamps):
                        # 转换回时间显示
                        dt = datetime.fromtimestamp(timestamps[idx])
                        
                        # 根据视图范围决定时间格式
                        view_range = self.price_plot.viewRange()
                        x_range = view_range[0]
                        view_width = x_range[1] - x_range[0]
                        
                        if view_width > 100:  # 显示范围较大时
                            # 显示年月
                            strings.append(dt.strftime('%Y-%m'))
                        else:
                            # 如果是新的一年或第一个刻度，显示完整日期
                            curr_year = dt.year
                            if prev_year != curr_year:
                                strings.append(dt.strftime('%Y-%m-%d'))
                                prev_year = curr_year
                            else:
                                # 否则只显示月日和时间
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
        base_index = len(self.current_data) - 1  # 基准索引（最新数据的位置）
        display_count = min(30, len(indices))  # 默认显示30根K线
        
        # 根据视图索引范围设置初始显示范围
        if hasattr(self, 'time_offset') and self.time_offset != 0:
            # 如果有时间偏移，使用偏移量计算显示范围
            center_idx = base_index + self.time_offset
            min_x = max(0, center_idx - display_count / 2)
            max_x = min(len(self.current_data) - 1, center_idx + display_count / 2)
        else:
            # 默认显示最后N根K线
            min_x = max(0, len(self.current_data) - display_count - 0.5)  # 确保不会小于0
            max_x = min(len(self.current_data) - 1, len(self.current_data) - 1 + 1.5)  # 确保不会超出范围
        
        # 设置初始显示范围
        self.price_plot.setXRange(min_x, max_x)
        print(f"初始显示范围: 索引 {min_x:.1f} -> {max_x:.1f}")
        print(f"显示 {display_count} 条K线，总计加载 {len(indices)} 条K线")
        
        # 打印详细的数据加载信息
        if len(self.current_data) > 0:
            loaded_start_time = datetime.fromtimestamp(self.current_data[0][0])
            loaded_end_time = datetime.fromtimestamp(self.current_data[-1][0])
            
            # 打印完整的时间范围信息
            if hasattr(self.parent(), 'full_time_range'):
                all_start_time = self.parent().full_time_range['start_time']
                all_end_time = self.parent().full_time_range['end_time']
                
                print("\n===== 数据时间范围详情 =====")
                print(f"总数据时间范围: {all_start_time.strftime('%Y-%m-%d %H:%M')} 至 {all_end_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"当前加载的数据: {loaded_start_time.strftime('%Y-%m-%d %H:%M')} 至 {loaded_end_time.strftime('%Y-%m-%d %H:%M')}")
                
                # 安全地获取初始显示范围的时间戳
                try:
                    start_idx = int(min_x)
                    end_idx = int(max_x)
                    if 0 <= start_idx < len(self.current_data) and 0 <= end_idx < len(self.current_data):
                        print(f"初始显示范围: {self.current_data[start_idx][0]:.0f} 至 {self.current_data[end_idx][0]:.0f} (时间戳)")
                except (IndexError, ValueError) as e:
                    print(f"计算初始显示范围时间戳时出错: {e}")
        
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

    def plot_current_data(self):
        """只绘制当前数据，不会触发递归调用
        
        与plot_kline不同，该方法不会设置full_data，只处理current_data
        """
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
        print(f"重新绘制K线范围：{start_time} 到 {end_time}")
        
        # 设置标题 - 添加全量数据的时间范围信息
        # 使用父组件中存储的完整时间范围信息
        if hasattr(self.parent(), 'full_time_range'):
            all_start_time = self.parent().full_time_range['start_time']
            all_end_time = self.parent().full_time_range['end_time']
            duration = all_end_time - all_start_time
            
            # 构建更详细的标题
            title_parts = [
                self.current_title,
                f"[{all_start_time.strftime('%Y-%m-%d')} 至 {all_end_time.strftime('%Y-%m-%d')}]",
                f"共 {len(self.full_data)} 条K线",
                f"跨度 {duration.days} 天 {duration.seconds // 3600} 小时"
            ]
            title_with_range = " | ".join(title_parts)
            self.price_plot.setTitle(title_with_range, color='#ffffff', size='12pt')
        else:
            # 如果没有完整时间范围信息，使用当前数据的时间范围
            all_start_time = start_time
            all_end_time = end_time
            duration = all_end_time - all_start_time
            title_with_range = f"{self.current_title} [{all_start_time.strftime('%Y-%m-%d')} 至 {all_end_time.strftime('%Y-%m-%d')}]"
            self.price_plot.setTitle(title_with_range, color='#ffffff', size='12pt')
        
        # 准备处理数据
        data_np = np.array(self.current_data, dtype=float)
        timestamps = data_np[:, 0].astype(np.int64)  # 实际时间戳，用于X轴标签
        opens = data_np[:, 1]
        highs = data_np[:, 2]
        lows = data_np[:, 3]
        closes = data_np[:, 4]
        volumes = data_np[:, 5]
        
        # 创建索引位置数组，用于绘制K线
        indices = np.arange(len(timestamps))
        
        # 创建时间戳到索引的映射，用于数据显示和查询
        self.ts_to_index = dict(zip(timestamps, indices))
        self.index_to_ts = dict(zip(indices, timestamps))
        
        # 生成新的K线数据 - 使用索引位置替代时间戳
        indexed_data = np.column_stack((indices, opens, highs, lows, closes, volumes))
        
        # 使用更高效的方式创建K线图元素
        try:
            print(f"重新生成K线图元素，K线宽度：{self.bar_width}")
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
                up_indices = indices[up_idx]
                up_volumes = volumes[up_idx] * 0.4
                
                up_volume_bar = pg.BarGraphItem(
                    x=up_indices, 
                    height=up_volumes, 
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
                down_indices = indices[down_idx]
                down_volumes = volumes[down_idx] * 0.4
                
                down_volume_bar = pg.BarGraphItem(
                    x=down_indices, 
                    height=down_volumes, 
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
            prev_year = None  # 跟踪前一个年份
            
            for value in values:
                try:
                    # 找到最接近的索引
                    idx = int(round(value))
                    if idx >= 0 and idx < len(timestamps):
                        # 转换回时间显示
                        dt = datetime.fromtimestamp(timestamps[idx])
                        
                        # 根据视图范围决定时间格式
                        view_range = self.price_plot.viewRange()
                        x_range = view_range[0]
                        view_width = x_range[1] - x_range[0]
                        
                        if view_width > 100:  # 显示范围较大时
                            # 显示年月
                            strings.append(dt.strftime('%Y-%m'))
                        else:
                            # 如果是新的一年或第一个刻度，显示完整日期
                            curr_year = dt.year
                            if prev_year != curr_year:
                                strings.append(dt.strftime('%Y-%m-%d'))
                                prev_year = curr_year
                            else:
                                # 否则只显示月日和时间
                                strings.append(dt.strftime('%m-%d %H:%M'))
                    else:
                        strings.append('')
                except:
                    strings.append('')
            return strings
        
        # 修改价格图和成交量图的X轴刻度函数
        self.price_plot.getAxis('bottom').tickStrings = tickStrings
        self.volume_plot.getAxis('bottom').tickStrings = tickStrings

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
        
        # 新增：当前数据日期范围显示
        self.date_range_label = QLabel("日期范围: 无数据")
        self.date_range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.date_range_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        
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
        nav_layout.addWidget(self.date_range_label)
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
        self.update_date_range_label()
    
    def update_date_range_label(self, visible_start_time=None, visible_end_time=None):
        """更新日期范围标签
        
        Args:
            visible_start_time: 可见数据的开始时间（可选）
            visible_end_time: 可见数据的结束时间（可选）
        """
        if not hasattr(self.kline_view, 'current_data') or not self.kline_view.current_data:
            self.date_range_label.setText("日期范围: 无数据")
            return
            
        # 获取当前显示数据的时间范围
        current_data = self.kline_view.current_data
        
        # 如果提供了可见时间范围，使用提供的值
        if visible_start_time and visible_end_time:
            start_time = visible_start_time
            end_time = visible_end_time
        else:
            # 否则使用当前数据的时间范围
            start_time = datetime.fromtimestamp(current_data[0][0])
            end_time = datetime.fromtimestamp(current_data[-1][0])
        
        # 获取全量数据的时间范围 - 使用父组件中存储的完整时间范围信息
        if hasattr(self, 'full_time_range'):
            all_start_time = self.full_time_range['start_time']
            all_end_time = self.full_time_range['end_time']
        else:
            # 如果没有完整时间范围信息，使用当前数据的时间范围
            all_start_time = start_time
            all_end_time = end_time
        
        # 设置标签文本
        range_text = f"当前: {start_time.strftime('%Y-%m-%d')} - {end_time.strftime('%Y-%m-%d')}  |  全部: {all_start_time.strftime('%Y-%m-%d')} - {all_end_time.strftime('%Y-%m-%d')}"
        self.date_range_label.setText(range_text)
    
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
    
    def load_data(self, symbol, interval, target_date=None):
        """加载特定币种和时间周期的数据
        
        Args:
            symbol: 币种代码，如ETHUSDT
            interval: 时间周期，如5m
            target_date: 如果指定，优先加载包含此日期的数据
            
        Returns:
            list: 加载的数据，每项为 [timestamp, open, high, low, close, volume]
        """
        # 检查是否已经加载过
        key = f"{symbol}_{interval}"
        if key in self.loaded_data and target_date is None:
            print(f"使用缓存数据: {key}, 数据长度: {len(self.loaded_data[key])}")
            return self.loaded_data[key]
            
        # 加载数据
        data_pattern = f"data/kline/{symbol}_{interval}_*.csv"
        print(f"\n===== 开始加载K线数据: {data_pattern} =====")
        
        # 列出匹配的文件
        import glob
        import re
        
        matching_files = sorted(glob.glob(data_pattern))
        if not matching_files:
            print(f"未找到数据文件: {data_pattern}")
            return []
            
        print(f"找到 {len(matching_files)} 个匹配的数据文件:")
        
        # 扫描所有文件的时间范围
        all_time_ranges = []
        for file in matching_files:
            try:
                # 使用pandas读取文件的第一行和最后一行
                df = pd.read_csv(file, nrows=1)  # 读取第一行
                if not df.empty:
                    # 确保时间戳列是字符串格式
                    first_ts_str = str(df.iloc[0]['timestamp'])
                    # 将时间字符串转换为datetime对象
                    first_dt = pd.to_datetime(first_ts_str)
                    first_ts = first_dt.timestamp()
                    
                    # 使用更高效的方式读取最后一行
                    with open(file, 'r', encoding='utf-8') as f:
                        # 跳过标题行
                        next(f)
                        # 读取最后一行
                        for line in f:
                            pass
                        last_line = line.strip()
                        
                        # 解析最后一行的时间戳
                        last_ts_str = last_line.split(',')[0]
                        last_dt = pd.to_datetime(last_ts_str)
                        last_ts = last_dt.timestamp()
                        
                        all_time_ranges.append((file, first_ts, last_ts))
                        print(f"  - {file}")
                        print(f"    时间范围: {first_dt.strftime('%Y-%m-%d %H:%M')} 至 {last_dt.strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                print(f"扫描文件 {file} 时间范围失败: {e}")
                continue
        
        if not all_time_ranges:
            print("未能获取任何文件的时间范围")
            return []
            
        # 计算完整的时间范围
        all_start_ts = min(ts[1] for ts in all_time_ranges)
        all_end_ts = max(ts[2] for ts in all_time_ranges)
        print(f"\n完整数据时间范围: {datetime.fromtimestamp(all_start_ts).strftime('%Y-%m-%d %H:%M')} 至 {datetime.fromtimestamp(all_end_ts).strftime('%Y-%m-%d %H:%M')}")
        
        # 如果指定了目标日期，优先加载该日期附近的数据
        if target_date:
            # 转换为datetime对象
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, '%Y-%m-%d')
            
            print(f"\n尝试加载目标日期 {target_date.strftime('%Y-%m-%d')} 附近的数据")
            
            # 找到包含目标日期的文件
            target_ts = target_date.timestamp()
            target_file = None
            for file, start_ts, end_ts in all_time_ranges:
                if start_ts <= target_ts <= end_ts:
                    target_file = file
                    print(f"找到包含目标日期的文件: {file}")
                    break
            
            if target_file:
                # 加载目标文件
                target_df = load_data_files(target_file)
                if not target_df.empty:
                    # 转换为列表格式
                    data_list = self._convert_df_to_list(target_df)
                    print(f"成功加载目标日期数据: {len(data_list)} 条记录")
                    
                    # 打印数据时间范围
                    if data_list:
                        start_time = datetime.fromtimestamp(data_list[0][0])
                        end_time = datetime.fromtimestamp(data_list[-1][0])
                        print(f"数据时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 至 {end_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    # 不缓存特定日期请求的数据
                    return data_list
        
        # 如果没有指定目标日期或者未找到合适文件，加载最新的数据
        print("\n加载所有数据文件...")
        all_data = []
        for file, start_ts, end_ts in all_time_ranges:
            try:
                print(f"\n加载文件: {file}")
                df = load_data_files(file)
                if not df.empty:
                    data_list = self._convert_df_to_list(df)
                    print(f"成功加载: {len(data_list)} 条记录")
                    all_data.extend(data_list)
                    
                    # 打印该文件的时间范围
                    if data_list:
                        start_time = datetime.fromtimestamp(data_list[0][0])
                        end_time = datetime.fromtimestamp(data_list[-1][0])
                        print(f"文件时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 至 {end_time.strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                print(f"加载文件失败: {str(e)}")
                continue
        
        if all_data:
            # 按时间戳排序
            all_data.sort(key=lambda x: x[0])
            print(f"\n总共加载了 {len(all_data)} 条记录")
            
            # 存储完整的时间范围信息
            self.full_time_range = {
                'start': all_start_ts,
                'end': all_end_ts,
                'start_time': datetime.fromtimestamp(all_start_ts),
                'end_time': datetime.fromtimestamp(all_end_ts)
            }
            
            # 缓存数据
            self.loaded_data[key] = all_data
            return all_data
        else:
            print("警告: 没有加载到任何数据")
            return []
        
    def _convert_df_to_list(self, df):
        """将DataFrame转换为K线数据列表"""
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
        return data_list
        
    def _print_data_stats(self, data_list):
        """打印数据统计信息"""
        if not data_list:
            return
            
        # 打印时间范围
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
        self.update_date_range_label()
    
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
        if not self.kline_view.full_data:
            print("没有可用数据进行跳转")
            return
        
        # 将Qt日期转换为Python日期
        selected_date = qt_date.toPyDate()
        
        # 将日期转换为datetime，设定为当天开始时间
        target_dt = datetime.combine(selected_date, datetime.min.time())
        target_ts = target_dt.timestamp()
        
        print(f"跳转到日期: {selected_date} (时间戳: {target_ts})")
        
        # 首先检查当前加载的数据是否包含目标日期
        current_data = self.kline_view.current_data
        full_data = self.kline_view.full_data
        
        # 获取完整数据的时间范围
        full_data_start_ts = full_data[0][0]
        full_data_end_ts = full_data[-1][0]
        
        # 检查目标日期是否在当前加载的数据范围内
        if full_data_start_ts <= target_ts <= full_data_end_ts:
            print("目标日期在当前加载的数据范围内")
            
            # 找到目标时间戳对应的索引
            timestamps = [item[0] for item in full_data]
            closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_ts))
            closest_time = datetime.fromtimestamp(timestamps[closest_idx])
            
            print(f"找到最接近的K线索引: {closest_idx} / {len(timestamps)-1}")
            print(f"对应时间: {closest_time}")
            
            # 计算显示范围 - 使用在full_data中的索引
            display_count = 60  # 显示更多K线
            
            # 确保索引不超出范围 - 目标日期在视图中央
            start_idx = max(0, closest_idx - display_count // 2)
            end_idx = min(len(full_data) - 1, start_idx + display_count - 1)
            
            # 如果靠近边界，调整显示窗口
            if end_idx >= len(full_data) - display_count // 4:
                # 靠近结束边界，调整起点
                start_idx = max(0, len(full_data) - display_count)
                end_idx = len(full_data) - 1
            elif start_idx <= display_count // 4:
                # 靠近起始边界，调整终点
                start_idx = 0
                end_idx = min(len(full_data) - 1, display_count - 1)
            
            # 更新当前数据以覆盖目标窗口
            self.kline_view.current_data = full_data[start_idx:end_idx + 1]
            
            # 重新绘制K线图
            self.plot_current_data()
            
            # 设置新的显示范围（使用平滑动画）
            self.setXRangeWithAnimation(0, len(self.current_data))
            
            # 更新时间偏移
            if closest_idx < len(full_data) - 1:
                self.kline_view.time_offset = len(full_data) - 1 - closest_idx
            else:
                self.kline_view.time_offset = 0
            
            print(f"设置显示范围: 0 -> {len(self.current_data)-1}，时间偏移: {self.kline_view.time_offset}")
            
        else:
            # 目标日期不在当前加载数据范围内，需要加载新数据
            print("目标日期不在当前加载的数据范围内，尝试加载包含该日期的数据")
            
            # 尝试加载特定日期的数据
            new_data = self.load_data(self.current_symbol, self.current_interval, selected_date)
            
            if new_data and len(new_data) > 0:
                # 检查新数据是否包含目标日期
                new_data_start_ts = new_data[0][0]
                new_data_end_ts = new_data[-1][0]
                
                if new_data_start_ts <= target_ts <= new_data_end_ts:
                    print("成功加载包含目标日期的数据")
                    
                    # 找到新数据中包含目标日期的位置
                    timestamps = [item[0] for item in new_data]
                    closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_ts))
                    closest_time = datetime.fromtimestamp(timestamps[closest_idx])
                    
                    print(f"找到最接近的K线索引: {closest_idx} / {len(timestamps)-1}")
                    print(f"对应时间: {closest_time}")
                    
                    # 更新图表数据
                    self.kline_view.full_data = new_data
                    display_count = min(len(new_data), 60)
                    
                    # 计算显示范围，使目标日期居中
                    start_idx = max(0, closest_idx - display_count // 2)
                    end_idx = min(len(new_data) - 1, start_idx + display_count - 1)
                    
                    # 更新当前显示数据
                    self.kline_view.current_data = new_data[start_idx:end_idx + 1]
                    
                    # 重新绘制K线图
                    title = f"{self.current_symbol} {INTERVAL_DISPLAY.get(self.current_interval, self.current_interval)} - {selected_date}"
                    self.kline_view.current_title = title
                    self.kline_view.plot_current_data()
                    
                    # 设置新的显示范围（使用平滑动画）
                    self.setXRangeWithAnimation(0, len(self.current_data))
                    
                    # 更新最新价格信息
                    self.update_price_info(new_data[closest_idx])
                else:
                    print(f"警告：加载的新数据不包含目标日期 {selected_date}")
                    QMessageBox.warning(self, "日期跳转", f"未找到包含 {selected_date} 的数据")
            else:
                print(f"无法加载包含日期 {selected_date} 的数据")
                QMessageBox.warning(self, "日期跳转", f"未找到包含 {selected_date} 的数据")
        
        # 更新信息显示
        self.update_position_label()
        self.update_date_range_label()
        
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

# 自定义Y轴，支持鼠标拖动缩放
class PriceYAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        self.view_box = kwargs.pop('view_box', None)
        super(PriceYAxisItem, self).__init__(*args, **kwargs)
        self.setLabel(text='价格', units=None)
        self.enableAutoSIPrefix(False)
        
        # 鼠标拖动相关变量
        self.mouse_dragging = False
        self.drag_start_pos = None
        self.drag_start_range = None
        
    def mousePressEvent(self, event):
        """处理鼠标按下事件，开始拖动"""
        if event.button() == Qt.MouseButton.LeftButton and self.view_box:
            self.mouse_dragging = True
            self.drag_start_pos = event.pos().y()
            self.drag_start_range = self.view_box.viewRange()[1]
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件，结束拖动"""
        if event.button() == Qt.MouseButton.LeftButton and self.mouse_dragging:
            self.mouse_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，执行Y轴缩放"""
        if self.mouse_dragging and self.view_box:
            current_y = event.pos().y()
            dy = current_y - self.drag_start_pos
            
            # 计算缩放系数，向下拖动时使范围变大（缩小）
            # 向上拖动时使范围变小（放大）
            scale_factor = 1.0 + dy * 0.01
            
            # 确保缩放系数在合理范围内
            scale_factor = max(0.5, min(scale_factor, 2.0))
            
            # 应用新的Y轴范围
            start_range = self.drag_start_range
            center = (start_range[0] + start_range[1]) / 2
            height = (start_range[1] - start_range[0]) * scale_factor
            
            # 计算新的Y轴范围
            min_y = center - height / 2
            max_y = center + height / 2
            
            # 设置新的视图范围
            self.view_box.setYRange(min_y, max_y, padding=0)
            event.accept()
        else:
            super().mouseMoveEvent(event)