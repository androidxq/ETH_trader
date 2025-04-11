"""
ETH交易系统主窗口

作为应用程序的主入口点，整合了所有功能：
- K线图显示
- 因子网格搜索
- 数据分析
- 因子策略交易
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
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QIcon, QAction

# 导入项目模块
from scripts.kline_view_pyqtgraph import KlineViewWidget
from scripts.grid_search_ui import GridSearchUI
from scripts.data_download_widget import DataDownloadWidget
from factor_strategy_ui import FactorStrategyUI  # 导入因子策略UI模块
from rl_strategies.rl_strategies_ui import RLStrategiesUI  # 导入强化学习策略UI模块

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
        
        # 创建因子策略UI标签页
        self.factor_strategy_tab = FactorStrategyUI()
        self.tab_widget.addTab(self.factor_strategy_tab, "因子策略")
        
        # 创建强化学习策略UI标签页
        self.rl_strategies_tab = RLStrategiesUI()
        self.tab_widget.addTab(self.rl_strategies_tab, "强化学习策略")
        
        # 添加调试代码，确认RL策略UI的默认设置
        print("DEBUG: 主窗口 - 检查RL策略UI默认设置")
        if hasattr(self.rl_strategies_tab, 'fee_spin'):
            print(f"DEBUG: 主窗口 - RL策略UI交易费率: {self.rl_strategies_tab.fee_spin.value()}")
        if hasattr(self.rl_strategies_tab, 'reward_type_combo'):
            print(f"DEBUG: 主窗口 - RL策略UI奖励类型: {self.rl_strategies_tab.reward_type_combo.currentText()}")
        
        # 创建网格搜索UI标签页
        self.grid_search_tab = GridSearchUI(self)  # 传入父组件
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
        
        # 连接标签页切换信号，确保UI元素正确更新
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # 连接因子策略的回测结果信号到K线图显示
        self.factor_strategy_tab.backtest_result_ready.connect(self.show_strategy_on_kline)
        
        # 连接获取K线数据的信号，允许因子策略模块使用K线图的数据
        self.factor_strategy_tab.request_kline_data.connect(self.get_kline_data)
        
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
        
        # 切换到因子策略动作
        factor_strategy_action = view_menu.addAction("因子策略")
        factor_strategy_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.factor_strategy_tab))
        
        # 切换到强化学习策略动作
        rl_strategies_action = view_menu.addAction("强化学习策略")
        rl_strategies_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.rl_strategies_tab))
        
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
                         "整合了K线图显示、因子网格搜索和因子策略功能\n"
                         "© 2025 ETH交易团队")
    
    def center_window(self):
        """将窗口居中显示"""
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.geometry()
        
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        
        self.move(x, y)

    def on_tab_changed(self, index):
        """处理标签页切换事件"""
        # 获取当前标签页
        current_tab = self.tab_widget.widget(index)
        
        # 如果是网格搜索标签页，确保其UI元素正确显示
        if current_tab == self.grid_search_tab:
            # 使用QTimer延迟执行，确保标签页完全显示后再刷新UI
            QTimer.singleShot(100, current_tab.update)
            
        # 如果是强化学习标签页，传递当前K线数据
        elif current_tab == self.rl_strategies_tab:
            # 获取K线数据并传递给强化学习UI
            kline_data = self.get_kline_data()
            if kline_data is not None:
                self.rl_strategies_tab.set_kline_data(kline_data)
                self.status_bar.showMessage(f"已为强化学习策略加载 {len(kline_data)} 条K线数据")
    
    def show_strategy_on_kline(self, strategy, result):
        """在K线图上显示策略回测结果
        
        参数:
            strategy (FactorTradingStrategy): 策略对象
            result (pd.DataFrame): 回测结果数据
        """
        try:
            # 切换到K线图标签页
            self.tab_widget.setCurrentWidget(self.kline_tab)
            
            # 显示回测结果
            self.kline_tab.add_trades_from_strategy(strategy)
            
            # 更新状态栏
            trade_count = strategy.metrics.get('trade_count', 0)
            win_rate = strategy.metrics.get('win_rate', 0)
            total_return = strategy.metrics.get('total_return', 0)
            
            self.status_bar.showMessage(
                f"因子策略回测完成: {trade_count}笔交易, 胜率{win_rate:.2f}%, 总收益{total_return:.2f}%"
            )
            
        except Exception as e:
            self.status_bar.showMessage(f"显示策略结果失败: {str(e)}")
            
    def get_kline_data(self):
        """从K线图组件获取当前加载的全量K线数据，供因子策略UI使用
        
        返回:
            pd.DataFrame: 包含完整K线数据的DataFrame，或None如果没有数据
        """
        if not hasattr(self, 'kline_tab'):
            self.status_bar.showMessage("K线图组件未初始化")
            return None
            
        # 调用K线图组件的方法获取数据
        data = self.kline_tab.get_full_kline_data()
        
        if data is not None:
            self.status_bar.showMessage(f"已从K线图获取 {len(data)} 条记录用于因子策略")
            
            # 将数据传递给因子策略UI
            if hasattr(self, 'factor_strategy_tab'):
                self.factor_strategy_tab.set_kline_data(data)
        else:
            self.status_bar.showMessage("未能从K线图获取数据，请先在K线图标签页加载数据")
            
        return data

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