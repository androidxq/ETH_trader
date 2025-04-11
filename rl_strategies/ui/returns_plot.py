"""
收益曲线图表组件 - PyQtGraph实现
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg

class ReturnsPlot(QWidget):
    """收益曲线图表组件"""
    
    def __init__(self, parent=None):
        """初始化组件"""
        super().__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建PyQtGraph绘图窗口
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # 设置图表样式
        self.plot_widget.setBackground('w')  # 白色背景
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)  # 显示网格并设置透明度
        
        # 设置标签
        self.plot_widget.setTitle('训练收益率曲线', color='k')
        self.plot_widget.setLabel('left', '资产价值', color='k')
        self.plot_widget.setLabel('bottom', '训练步数', color='k')
        
        # 创建曲线对象
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2))
        
        # 存储数据
        self.returns = []
        self.steps = []
        
        # 自动范围标志
        self.auto_range = True
        
        # 配置参数
        self.window_size = 1000  # 显示窗口大小
        self.max_points = 200    # 最大渲染点数
    
    def update_plot(self, returns, steps=None):
        """更新收益率曲线
        
        参数:
            returns: 收益率列表
            steps: 步数列表，如果为None则自动生成
        """
        if not returns:
            return
            
        # 更新数据缓存
        self.returns = returns
        
        # 如果没有提供步数，使用索引
        if steps is None:
            # 不再使用简单的索引，而是使用None表示需要外部提供实际步数
            raise ValueError("必须提供实际的训练步数(K线数量)作为steps参数")
        else:
            self.steps = steps
        
        # 窗口化处理
        if len(self.returns) > self.window_size:
            start_idx = len(self.returns) - self.window_size
            display_returns = self.returns[start_idx:]
            display_steps = self.steps[start_idx:]
        else:
            display_returns = self.returns
            display_steps = self.steps
            
        # 采样处理
        if len(display_returns) > self.max_points:
            # 使用实际步数范围进行采样
            min_step = min(display_steps)
            max_step = max(display_steps)
            sample_steps = np.linspace(min_step, max_step, self.max_points)
            
            # 对收益率进行插值采样
            display_returns = np.interp(
                sample_steps,
                display_steps,
                display_returns
            ).tolist()
            display_steps = sample_steps.tolist()
        
        # 更新曲线数据
        self.curve.setData(x=display_steps, y=display_returns, skipFiniteCheck=True)
        
        # 自动调整Y轴范围
        if len(display_returns) > 1:
            y_min = min(display_returns)
            y_max = max(display_returns)
            y_range = y_max - y_min
            self.plot_widget.setYRange(
                y_min - y_range * 0.1,
                y_max + y_range * 0.1
            )
    
    def set_auto_range(self, enabled):
        """设置是否启用自动范围"""
        self.auto_range = enabled
        if enabled:
            self.plot_widget.enableAutoRange()
    
    def clear(self):
        """清除图表数据"""
        self.returns = []
        self.steps = []
        self.curve.clear()