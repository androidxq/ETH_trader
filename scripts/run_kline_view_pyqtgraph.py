"""
PyQtGraph K线图独立运行脚本
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入PyQt6
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# 导入新的PyQtGraph K线图模块
from scripts.kline_view_pyqtgraph import KlineViewWidget

if __name__ == "__main__":
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建K线图窗口
    kline_widget = KlineViewWidget()
    kline_widget.setWindowTitle("PyQtGraph K线图显示")
    kline_widget.resize(1280, 800)  # 设置窗口大小
    kline_widget.show()
    
    # 运行应用程序
    sys.exit(app.exec()) 