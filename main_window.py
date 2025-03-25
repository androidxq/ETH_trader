"""
ETH交易系统主窗口

作为应用程序的主入口点，整合了所有功能：
- K线图显示
- 因子网格搜索
- 数据分析
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                           QStatusBar, QComboBox, QSplitter, QMenuBar, QMessageBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QAction

# 导入项目模块
from scripts.kline_view_pyqtgraph import KlineViewWidget
from scripts.grid_search_ui import GridSearchUI
from scripts.data_download_widget import DataDownloadWidget

class MainWindow(QMainWindow):
    """ETH交易系统主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ETH交易系统")
        self.setMinimumSize(1280, 800)  # 设置窗口大小
        self.init_ui()
        self.create_menu()
        self.center_window()
        
    def init_ui(self):
        """初始化UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 创建K线图标签页
        self.kline_tab = KlineViewWidget()
        self.tab_widget.addTab(self.kline_tab, "K线图")
        
        # 创建网格搜索UI标签页
        self.grid_search_tab = GridSearchUI()
        self.tab_widget.addTab(self.grid_search_tab, "因子网格搜索")
        
        # 添加数据下载标签页
        self.data_download_tab = DataDownloadWidget()
        self.tab_widget.addTab(self.data_download_tab, "数据下载")
        
        # 添加标签页到主布局
        main_layout.addWidget(self.tab_widget)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("ETH交易系统启动完成")
        
    def create_menu(self):
        """创建菜单栏"""
        menu_bar = self.menuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu("文件")
        
        # 退出动作
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)
        
        # 视图菜单
        view_menu = menu_bar.addMenu("视图")
        
        # 切换到K线图动作
        kline_action = view_menu.addAction("K线图")
        kline_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.kline_tab))
        
        # 切换到网格搜索动作
        grid_search_action = view_menu.addAction("因子网格搜索")
        grid_search_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.grid_search_tab))
        
        # 添加数据下载动作
        data_download_action = view_menu.addAction("数据下载")
        data_download_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.data_download_tab))
        
        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")
        
        # 关于动作
        about_action = help_menu.addAction("关于")
        about_action.triggered.connect(self.show_about)
        
    def show_about(self):
        """显示关于信息"""
        QMessageBox.about(self, "关于ETH交易系统", 
                         "ETH交易系统 v0.3.0\n\n"
                         "整合了K线图显示和因子网格搜索功能\n"
                         "© 2025 ETH交易团队")
    
    def center_window(self):
        """将窗口居中显示"""
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.geometry()
        
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        
        self.move(x, y)

if __name__ == "__main__":
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    main_window = MainWindow()
    main_window.show()
    
    # 运行应用程序
    sys.exit(app.exec()) 