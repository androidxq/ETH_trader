"""学习率曲线图实现

使用PyQtGraph实现学习率变化曲线的可视化
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt

class LearningRatePlot(QWidget):
    """学习率曲线图组件"""
    
    def __init__(self, parent=None):
        """初始化学习率曲线图"""
        super().__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建绘图窗口
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # 设置图表样式
        self.plot_widget.setBackground('w')  # 白色背景
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # 设置标签
        self.plot_widget.setLabel('left', '学习率')
        self.plot_widget.setLabel('bottom', '训练步数')
        self.plot_widget.setTitle('学习率变化曲线', color='k', size='12pt')
        
        # 创建曲线对象
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='r', width=2))
        
        # 数据缓存
        self.learning_rates = []
        self.steps = []
        
        # 配置参数
        self.window_size = 1000  # 显示窗口大小
        self.max_points = 200    # 最大渲染点数
        
    def update_plot(self, learning_rates, steps=None):
        """更新学习率曲线
        
        参数:
            learning_rates: 学习率数据列表
            steps: 对应的步数列表，如果为None则使用索引作为步数
        """
        if not learning_rates:
            return
            
        # 更新数据缓存
        self.learning_rates = learning_rates
        
        # 如果没有提供步数，使用索引
        if steps is None:
            self.steps = list(range(len(learning_rates)))
        else:
            self.steps = steps
            
        # 窗口化处理
        if len(self.learning_rates) > self.window_size:
            start_idx = len(self.learning_rates) - self.window_size
            display_rates = self.learning_rates[start_idx:]
            display_steps = self.steps[start_idx:]
        else:
            display_rates = self.learning_rates
            display_steps = self.steps
            
        # 采样处理
        if len(display_rates) > self.max_points:
            indices = np.linspace(0, len(display_rates)-1, self.max_points, dtype=int)
            display_rates = [display_rates[i] for i in indices]
            display_steps = [display_steps[i] for i in indices]
            
        # 更新曲线数据
        self.curve.setData(display_steps, display_rates)
        
        # 自动调整Y轴范围
        if len(display_rates) > 1:
            y_min = min(display_rates)
            y_max = max(display_rates)
            y_range = y_max - y_min
            self.plot_widget.setYRange(
                y_min - y_range * 0.1,
                y_max + y_range * 0.1
            )
        
    def clear(self):
        """清空图表数据"""
        self.learning_rates = []
        self.steps = []
        self.curve.setData([], [])