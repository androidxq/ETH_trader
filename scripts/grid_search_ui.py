"""
网格搜索UI界面

提供可视化界面监控网格搜索进度
"""

import os
import sys
from pathlib import Path
import pickle
import time
from datetime import datetime
import threading
import multiprocessing as mp
import psutil  # 用于获取系统资源使用情况

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QProgressBar, QPushButton, 
                           QTextEdit, QGroupBox, QGridLayout, QTableWidget, 
                           QTableWidgetItem, QTabWidget, QSplitter, QFileDialog,
                           QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette

# 导入项目模块
from scripts.grid_search_factors import FactorGridSearch
from factor_research.config.grid_search_config import PARAM_GRID, SPECIAL_COMBINATIONS, FIXED_PARAMS
from scripts.kline_view import KlineViewWidget


# 定义可以被pickle的函数（移到类外部）
def wrapped_single_search(self, params, data):
    """添加进度更新的包装方法"""
    if not self.running:
        return None
        
    # 发送开始处理信号
    self.update_signal.emit({
        'type': 'param_start',
        'params': params
    })
    
    # 调用原方法
    result = self.searcher._single_search(params, data)
    
    # 发送完成处理信号
    self.update_signal.emit({
        'type': 'param_end',
        'result': result
    })
    
    return result


# 定义一个辅助函数用于处理单个参数组合的搜索
def process_param_search(param_data_tuple):
    """处理单个参数组合的搜索，解决序列化问题
    
    Args:
        param_data_tuple: 包含参数和数据的元组 (params, data)
        
    Returns:
        搜索结果字典
    """
    try:
        from scripts.grid_search_factors import FactorGridSearch
        searcher = FactorGridSearch()
        params, data = param_data_tuple
        result = searcher._single_search(params, data)
        
        # 清理内存
        import gc
        gc.collect()
        
        # 清理searcher对象
        del searcher
        
        return result
    except Exception as e:
        print(f"处理参数组合时出错: {str(e)}")
        return None
    finally:
        # 确保在任何情况下都清理内存
        import gc
        gc.collect()


class GridSearchWorker(QThread):
    """后台执行网格搜索的工作线程"""
    update_signal = pyqtSignal(dict)  # 发送进度更新信号
    finished_signal = pyqtSignal()    # 发送完成信号
    
    def __init__(self):
        super().__init__()
        self.searcher = None
        self.running = False
        self.paused = False
        self.current_process_pool = None
        self.active_process_ids = set()  # 存储活跃进程ID
        
    def run(self):
        """执行网格搜索"""
        self.running = True
        self.searcher = FactorGridSearch()
        
        try:
            # 加载数据和检查点
            completed_results, completed_params = self.searcher._find_latest_checkpoint()
            data = self.searcher._load_data()
            
            if data.empty:
                self.update_signal.emit({
                    'type': 'error',
                    'message': "错误: 未能加载数据，无法执行网格搜索"
                })
                return
                
            # 创建参数组合列表
            all_param_combinations = []
            
            # 从配置中生成网格参数组合
            for forward_period in PARAM_GRID.get("forward_period", [12]):
                for generations in PARAM_GRID.get("generations", [100]):
                    for population_size in PARAM_GRID.get("population_size", [1000]):
                        for tournament_size in PARAM_GRID.get("tournament_size", [20]):
                            # 合并固定参数和可变参数
                            params = {
                                "forward_period": forward_period,
                                "generations": generations,
                                "population_size": population_size,
                                "tournament_size": tournament_size,
                                **FIXED_PARAMS
                            }
                            all_param_combinations.append(params)
            
            # 添加特殊组合
            for special_combo in SPECIAL_COMBINATIONS:
                # 确保特殊组合中包含所有必要的固定参数
                combo = {**FIXED_PARAMS, **special_combo}
                all_param_combinations.append(combo)
                
            total_all_combinations = len(all_param_combinations)
            
            # 过滤出尚未完成的参数组合
            if completed_params:
                param_combinations = []
                for params in all_param_combinations:
                    # 创建与已完成参数集合中相同格式的标识
                    param_tuple = tuple(sorted([
                        (k, str(v)) for k, v in params.items()
                    ]))
                    if param_tuple not in completed_params:
                        param_combinations.append(params)
                
                self.update_signal.emit({
                    'type': 'progress_init',
                    'total': total_all_combinations,
                    'completed': len(completed_params),
                    'remaining': len(param_combinations),
                    'message': f"从断点继续执行: 总共 {total_all_combinations} 个组合，已完成 {len(completed_params)} 个，剩余 {len(param_combinations)} 个"
                })
            else:
                param_combinations = all_param_combinations
                self.update_signal.emit({
                    'type': 'progress_init',
                    'total': total_all_combinations,
                    'completed': 0,
                    'remaining': len(param_combinations),
                    'message': f"从头开始执行: 共 {total_all_combinations} 个参数组合"
                })
                
            # 如果所有组合都已完成，直接生成报告并返回
            if not param_combinations:
                self.update_signal.emit({
                    'type': 'complete',
                    'message': "所有参数组合已完成，直接生成报告"
                })
                self.searcher._generate_report(completed_results)
                return
                
            # 准备任务数据元组
            process_args = [(params, data) for params in param_combinations]
            
            # 分批处理参数设置
            total_combinations = len(param_combinations)
            batch_size = 10  # 每批处理10个组合
            num_batches = (total_combinations + batch_size - 1) // batch_size
            
            # 设置进程数为CPU核心数的一半
            num_processes = max(1, mp.cpu_count() // 2)
            
            self.update_signal.emit({
                'type': 'batch_info',
                'total_batches': num_batches,
                'processes': num_processes,
                'process_ids': list(self.active_process_ids),  # 发送当前进程ID列表
                'message': f"总共 {total_combinations} 个参数组合，分 {num_batches} 批处理，使用 {num_processes} 个进程"
            })
            
            # 使用多进程执行搜索
            results = list(completed_results)  # 从已完成的结果开始
            
            # 确保正确设置多进程启动方法
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # 如果已经设置过，会抛出RuntimeError
                pass
                
            # 分批处理
            for batch_idx in range(num_batches):
                # 检查是否应该停止
                if not self.running:
                    self.update_signal.emit({
                        'type': 'stopped',
                        'message': "搜索已停止"
                    })
                    return
                    
                # 检查是否暂停
                while self.paused and self.running:
                    time.sleep(0.5)
                
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_combinations)
                batch_args = process_args[start_idx:end_idx]
                
                self.update_signal.emit({
                    'type': 'batch_start',
                    'batch_idx': batch_idx,
                    'total_batches': num_batches,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'message': f"正在处理第 {batch_idx+1}/{num_batches} 批 (组合 {start_idx+1} 到 {end_idx})"
                })
                
                batch_results = []
                
                # 使用多进程处理参数组合
                with mp.Pool(processes=num_processes) as pool:
                    self.current_process_pool = pool
                    
                    # 获取进程池中的所有进程ID
                    self.active_process_ids = {p.pid for p in pool._pool}
                    self.update_signal.emit({
                        'type': 'process_update',
                        'process_ids': list(self.active_process_ids)
                    })
                    
                    # 使用imap处理结果并实时获取
                    for i, result in enumerate(pool.imap(process_param_search, batch_args)):
                        # 检查是否应该停止
                        if not self.running:
                            pool.terminate()  # 立即终止所有进程
                            self.update_signal.emit({
                                'type': 'stopped',
                                'message': "搜索已停止"
                            })
                            return
                            
                        # 检查是否暂停 (注意：这里只能暂停获取结果，无法暂停已经在运行的进程)
                        while self.paused and self.running:
                            time.sleep(0.5)
                        
                        # 处理结果
                        if result is not None:  # 只添加有效结果
                            batch_results.append(result)
                            results.append(result)
                            
                            # 实时保存结果
                            with open(self.searcher.results_file, 'wb') as f:
                                pickle.dump(results, f)
                        
                        # 发送进度更新
                        self.update_signal.emit({
                            'type': 'batch_progress',
                            'batch_idx': batch_idx,
                            'batch_progress': (i + 1) / len(batch_args),
                            'total_progress': (len(results) - len(completed_results)) / total_combinations,
                            'overall_progress': len(results) / total_all_combinations,
                            'current_combination': start_idx + i + 1,
                            'result': result
                        })
                        
                        # 更新进程信息
                        self.update_signal.emit({
                            'type': 'process_update',
                            'process_ids': list(self.active_process_ids)
                        })
                        
                        # 每处理完一个结果就清理内存
                        import gc
                        gc.collect()
                
                # 进程池完成后清空进程ID集合
                self.active_process_ids.clear()
                
                # 保存批次中间结果
                intermediate_file = f"{self.searcher.results_dir}/grid_search_intermediate_batch_{batch_idx+1}_{num_batches}_{self.searcher.timestamp}.pkl"
                with open(intermediate_file, 'wb') as f:
                    pickle.dump(results, f)
                    
                self.update_signal.emit({
                    'type': 'batch_end',
                    'batch_idx': batch_idx,
                    'total_batches': num_batches,
                    'message': f"已完成批次 {batch_idx+1}/{num_batches} 的处理"
                })
                
                # 每批次结束后强制清理内存
                import gc
                gc.collect()
                
                # 清理不再需要的数据
                del batch_results
                gc.collect()
            
            # 所有搜索结束后，生成一次最终报告
            self.update_signal.emit({
                'type': 'generating_report',
                'message': "所有参数组合搜索完成，开始生成最终报告..."
            })
            
            self.searcher._generate_report(results)
            
            self.update_signal.emit({
                'type': 'complete',
                'message': f"网格搜索完成! 结果已保存到: {self.searcher.results_file}，最终报告已生成: {self.searcher.report_file}"
            })
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            self.update_signal.emit({
                'type': 'error',
                'message': f"执行网格搜索时出错: {str(e)}\n{traceback_str}"
            })
        
        finally:
            # 确保清理所有资源
            self.running = False
            if self.current_process_pool is not None:
                self.current_process_pool.terminate()
            import gc
            gc.collect()
            self.finished_signal.emit()
        
    def stop(self):
        """停止搜索"""
        self.running = False
        # 如果有正在运行的进程池，立即终止它
        if self.current_process_pool is not None:
            self.current_process_pool.terminate()
        
    def pause(self):
        """暂停搜索"""
        self.paused = True
        
    def resume(self):
        """恢复搜索"""
        self.paused = False

class GridSearchUI(QMainWindow):
    """网格搜索UI界面"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_batch = 0
        self.total_batches = 0
        self.total_combinations = 0
        self.completed_combinations = 0
        self.factors_found = []  # 存储找到的因子
        self.active_processes = 0  # 当前活跃进程数
        self.process_info = {}  # 存储进程信息
        
        # 创建定时器更新内存使用情况
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.setInterval(5000)  # 5秒更新一次
        
        # 创建定时器更新进程资源使用情况
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.update_process_resources)
        self.process_timer.setInterval(1000)  # 1秒更新一次
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("ETH因子网格搜索")
        self.setMinimumSize(1200, 800)  # 增加窗口默认大小
        
        # 设置Windows 11风格的边距和间距，调整颜色对比度
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d3d3d3;
                border-radius: 8px;
                padding-top: 16px;
                margin-top: 8px;
                background-color: white;
                color: #000000;  /* 黑色文字 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #000000;  /* 黑色文字 */
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1683d8;
            }
            QPushButton:pressed {
                background-color: #006cc1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QProgressBar {
                border: 1px solid #d3d3d3;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
                color: #000000;  /* 黑色文字 */
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #d3d3d3;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #d3d3d3;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
                color: #000000;  /* 黑色文字 */
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
                font-weight: bold;
            }
            QTableWidget {
                border: 1px solid #d3d3d3;
                border-radius: 4px;
                gridline-color: #f0f0f0;
                color: #000000;  /* 黑色文字 */
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d3d3d3;
                border-left: none;
                color: #000000;  /* 黑色文字 */
                font-weight: bold;
            }
            QLabel {
                color: #000000;  /* 黑色文字 */
            }
            QTextEdit {
                color: #000000;  /* 黑色文字 */
                background-color: white;
                border: 1px solid #d3d3d3;
                border-radius: 4px;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # ======= 创建主标签页 =======
        main_tabs = QTabWidget()
        main_tabs.setMinimumHeight(650)  # 确保标签页有足够的高度
        
        # ======= 1. 网格搜索状态标签页 =======
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        status_layout.setContentsMargins(15, 15, 15, 15)
        status_layout.setSpacing(15)
        
        # 创建状态组
        status_group = QGroupBox("网格搜索状态")
        status_inner_layout = QVBoxLayout(status_group)
        status_inner_layout.setContentsMargins(15, 25, 15, 15)
        status_inner_layout.setSpacing(20)
        
        # 总进度
        total_progress_layout = QHBoxLayout()
        self.total_progress_label = QLabel("总进度: 0/0 (0%)")
        self.total_progress_bar = QProgressBar()
        self.total_progress_bar.setRange(0, 100)
        self.total_progress_bar.setValue(0)
        self.total_progress_bar.setMinimumHeight(25)  # 增加进度条高度
        total_progress_layout.addWidget(self.total_progress_label, 3)
        total_progress_layout.addWidget(self.total_progress_bar, 7)
        status_inner_layout.addLayout(total_progress_layout)
        
        # 当前批次进度
        batch_progress_layout = QHBoxLayout()
        self.batch_progress_label = QLabel("当前批次: 0/0 (0%)")
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setMinimumHeight(25)  # 增加进度条高度
        batch_progress_layout.addWidget(self.batch_progress_label, 3)
        batch_progress_layout.addWidget(self.batch_progress_bar, 7)
        status_inner_layout.addLayout(batch_progress_layout)
        
        # 资源使用情况
        resource_layout = QHBoxLayout()
        self.process_label = QLabel("活跃进程数: 0")
        self.memory_label = QLabel("内存使用率: 0%")
        font = self.process_label.font()
        font.setBold(True)
        font.setPointSize(11)  # 增加字体大小
        self.process_label.setFont(font)
        self.memory_label.setFont(font)
        resource_layout.addWidget(self.process_label)
        resource_layout.addSpacing(50)
        resource_layout.addWidget(self.memory_label)
        resource_layout.addStretch()
        status_inner_layout.addLayout(resource_layout)
        
        # 状态信息
        status_info_layout = QHBoxLayout()
        self.status_label = QLabel("状态: 等待开始")
        status_font = self.status_label.font()
        status_font.setBold(True)
        status_font.setPointSize(11)  # 增加字体大小
        self.status_label.setFont(status_font)
        status_info_layout.addWidget(self.status_label)
        status_info_layout.addStretch()
        status_inner_layout.addLayout(status_info_layout)
        
        # 添加到状态标签页
        status_layout.addWidget(status_group)
        
        # 添加一个简单的状态日志
        status_log_group = QGroupBox("状态日志")
        status_log_layout = QVBoxLayout(status_log_group)
        status_log_layout.setContentsMargins(15, 25, 15, 15)
        
        self.status_log_text = QTextEdit()
        self.status_log_text.setReadOnly(True)
        self.status_log_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        status_log_layout.addWidget(self.status_log_text)
        
        status_layout.addWidget(status_log_group)
        
        main_tabs.addTab(status_tab, "搜索状态")
        
        # ======= 2. 进程信息标签页 =======
        process_tab = QWidget()
        process_layout = QVBoxLayout(process_tab)
        process_layout.setContentsMargins(15, 15, 15, 15)
        process_layout.setSpacing(15)
        
        process_info_group = QGroupBox("进程信息")
        process_info_layout = QVBoxLayout(process_info_group)
        process_info_layout.setContentsMargins(15, 25, 15, 15)
        
        # 创建进程信息表格
        self.process_table = QTableWidget(0, 3)
        self.process_table.setHorizontalHeaderLabels(["进程ID", "CPU使用率", "内存使用率"])
        
        # 设置表格列宽和行高
        header = self.process_table.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeMode.Fixed)
        header.setSectionResizeMode(1, header.ResizeMode.Fixed)
        header.setSectionResizeMode(2, header.ResizeMode.Stretch)
        self.process_table.setColumnWidth(0, 200)
        self.process_table.setColumnWidth(1, 200)
        self.process_table.verticalHeader().setDefaultSectionSize(35)  # 增加行高
        
        process_info_layout.addWidget(self.process_table)
        process_layout.addWidget(process_info_group)
        
        main_tabs.addTab(process_tab, "进程信息")
        
        # ======= 3. 执行情况标签页 =======
        execution_tab = QWidget()
        execution_layout = QVBoxLayout(execution_tab)
        execution_layout.setContentsMargins(15, 15, 15, 15)
        execution_layout.setSpacing(15)
        
        # 当前参数组合
        params_group = QGroupBox("当前处理的参数组合")
        params_grid = QGridLayout(params_group)
        params_grid.setContentsMargins(15, 25, 15, 15)  # 增加内边距
        params_grid.setVerticalSpacing(20)  # 增加垂直间距
        params_grid.setColumnStretch(0, 1)
        params_grid.setColumnStretch(1, 3)
        
        param_labels = [
            ("预测周期", "forward_period"),
            ("种群大小", "population_size"),
            ("进化代数", "generations"),
            ("锦标赛大小", "tournament_size"),
            ("交叉概率", "p_crossover"),
            ("子树变异概率", "p_subtree_mutation"),
            ("提升变异概率", "p_hoist_mutation"),
            ("点变异概率", "p_point_mutation"),
            ("复杂度惩罚", "parsimony_coefficient")
        ]
        
        self.param_value_labels = {}
        
        for i, (label_text, param_name) in enumerate(param_labels):
            label = QLabel(f"{label_text}:")
            value_label = QLabel("-")
            
            # 设置更大的字体和最小高度
            font = label.font()
            font.setPointSize(11)  # 增加字体大小
            label.setFont(font)
            value_label.setFont(font)
            
            # 设置最小高度确保完整显示
            label.setMinimumHeight(30)
            value_label.setMinimumHeight(30)
            
            params_grid.addWidget(label, i, 0)
            params_grid.addWidget(value_label, i, 1)
            self.param_value_labels[param_name] = value_label
            
        execution_layout.addWidget(params_group)
        
        # 日志输出
        log_group = QGroupBox("执行日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(15, 25, 15, 15)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        font = self.log_text.font()
        font.setPointSize(10)  # 增加日志字体大小
        self.log_text.setFont(font)
        log_layout.addWidget(self.log_text)
        
        execution_layout.addWidget(log_group)
        
        main_tabs.addTab(execution_tab, "执行情况")
        
        # ======= 4. 因子结果标签页 =======
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(15, 15, 15, 15)
        results_layout.setSpacing(15)
        
        # 因子结果表格
        results_group = QGroupBox("已找到的因子")
        results_inner_layout = QVBoxLayout(results_group)
        results_inner_layout.setContentsMargins(15, 25, 15, 15)
        
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels([
            "预测周期", "表达式", "IC值", "稳定性", "做多收益", "做空收益"
        ])
        
        # 设置表格列宽和行高
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeMode.Fixed)
        header.setSectionResizeMode(1, header.ResizeMode.Stretch)
        header.setSectionResizeMode(2, header.ResizeMode.Fixed)
        header.setSectionResizeMode(3, header.ResizeMode.Fixed)
        header.setSectionResizeMode(4, header.ResizeMode.Fixed)
        header.setSectionResizeMode(5, header.ResizeMode.Fixed)
        self.results_table.setColumnWidth(0, 100)
        self.results_table.setColumnWidth(2, 100)
        self.results_table.setColumnWidth(3, 100)
        self.results_table.setColumnWidth(4, 100)
        self.results_table.setColumnWidth(5, 100)
        self.results_table.verticalHeader().setDefaultSectionSize(35)  # 增加行高
        
        results_inner_layout.addWidget(self.results_table)
        results_layout.addWidget(results_group)
        
        main_tabs.addTab(results_tab, "已找到的因子")
        
        # ======= 5. K线图标签页 =======
        kline_tab = QWidget()
        kline_layout = QVBoxLayout(kline_tab)
        kline_layout.setContentsMargins(15, 15, 15, 15)
        kline_layout.setSpacing(0)
        
        # 添加K线图组件
        self.kline_widget = KlineViewWidget()
        kline_layout.addWidget(self.kline_widget)
        
        main_tabs.addTab(kline_tab, "K线图")
        
        # 添加主标签页到布局
        main_layout.addWidget(main_tabs)
        
        # ======= 底部按钮区域 =======
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(15)
        
        self.start_button = QPushButton("开始搜索")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.report_button = QPushButton("打开报告")
        
        # 增大按钮尺寸
        button_font = self.start_button.font()
        button_font.setPointSize(11)
        button_font.setBold(True)
        self.start_button.setFont(button_font)
        self.pause_button.setFont(button_font)
        self.stop_button.setFont(button_font)
        self.report_button.setFont(button_font)
        
        self.start_button.setMinimumHeight(40)
        self.pause_button.setMinimumHeight(40)
        self.stop_button.setMinimumHeight(40)
        self.report_button.setMinimumHeight(40)
        
        self.start_button.clicked.connect(self.start_search)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_search)
        self.report_button.clicked.connect(self.open_report)
        
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.report_button.setEnabled(False)
        
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.report_button)
        buttons_layout.addStretch()
        
        main_layout.addLayout(buttons_layout)
        
        # 显示窗口
        self.center_window()
        self.show()
        
    def center_window(self):
        """将窗口居中显示"""
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.geometry()
        
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        
        self.move(x, y)
        
    def update_memory_usage(self):
        """更新内存使用情况"""
        try:
            memory_percent = psutil.virtual_memory().percent
            self.memory_label.setText(f"内存使用率: {memory_percent:.1f}%")
        except Exception as e:
            print(f"获取内存使用率失败: {str(e)}")
            
    def start_search(self):
        """开始网格搜索"""
        if self.worker is not None and self.worker.isRunning():
            return
            
        # 清空表格和日志
        self.log_text.clear()
        self.results_table.setRowCount(0)
        self.factors_found = []
        
        # 重置进度条
        self.total_progress_bar.setValue(0)
        self.batch_progress_bar.setValue(0)
        
        # 更新状态
        self.status_label.setText("状态: 正在搜索...")
        self.log_message("开始网格搜索...")
        
        # 更新按钮状态
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.report_button.setEnabled(False)
        
        # 创建并启动worker线程
        self.worker = GridSearchWorker()
        self.worker.update_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.search_finished)
        self.worker.start()
        
        # 启动监控
        self.memory_timer.start()
        self.process_timer.start()
        
    def toggle_pause(self):
        """暂停/恢复搜索"""
        if self.worker is None or not self.worker.isRunning():
            return
            
        if self.worker.paused:
            # 恢复
            self.worker.resume()
            self.pause_button.setText("暂停")
            self.status_label.setText("状态: 正在搜索...")
            self.log_message("恢复搜索...")
        else:
            # 暂停
            self.worker.pause()
            self.pause_button.setText("继续")
            self.status_label.setText("状态: 已暂停")
            self.log_message("暂停搜索...")
        
    def stop_search(self):
        """停止搜索"""
        if self.worker is None or not self.worker.isRunning():
            return
            
        reply = QMessageBox.question(
            self, 
            "确认停止", 
            "确定要停止搜索吗？已完成的结果会被保存。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.worker.stop()
            self.status_label.setText("状态: 正在停止...")
            self.log_message("正在停止搜索...")
        
    def search_finished(self):
        """搜索完成处理"""
        # 更新按钮状态
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.report_button.setEnabled(True)
        
        # 停止监控
        self.memory_timer.stop()
        self.process_timer.stop()
        
        # 更新状态
        self.status_label.setText("状态: 已完成")
        self.log_message("网格搜索完成!")
        
        # 重置进程数
        self.active_processes = 0
        self.process_label.setText(f"活跃进程数: {self.active_processes}")
        
    def open_report(self):
        """打开报告文件"""
        report_dir = Path("results/grid_search")
        if not report_dir.exists():
            QMessageBox.warning(self, "警告", "报告目录不存在")
            return
            
        # 查找最新的报告文件
        report_files = list(report_dir.glob("grid_search_report_*.md"))
        if not report_files:
            QMessageBox.warning(self, "警告", "未找到报告文件")
            return
            
        # 按修改时间排序，获取最新的报告
        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_report = report_files[0]
        
        # 使用系统默认应用打开报告
        import os
        import platform
        
        if platform.system() == 'Windows':
            os.startfile(latest_report)
        elif platform.system() == 'Darwin':  # macOS
            os.system(f'open "{latest_report}"')
        else:  # Linux
            os.system(f'xdg-open "{latest_report}"')
            
    def update_process_resources(self):
        """更新进程资源使用情况"""
        if not self.process_info:
            return
            
        for i, (pid, info) in enumerate(self.process_info.items()):
            try:
                process = psutil.Process(pid)
                cpu_percent = process.cpu_percent()
                memory_percent = process.memory_percent()
                
                # 更新表格中的资源使用情况
                cpu_item = QTableWidgetItem(f"{cpu_percent:.1f}%")
                mem_item = QTableWidgetItem(f"{memory_percent:.1f}%")
                
                # 设置文本对齐
                cpu_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                mem_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                self.process_table.setItem(i, 1, cpu_item)
                self.process_table.setItem(i, 2, mem_item)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.process_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self.process_table.setItem(i, 2, QTableWidgetItem("N/A"))
                
    def update_process_info(self, process_ids):
        """更新进程信息显示"""
        self.process_info = {}
        self.process_table.setRowCount(len(process_ids))
        
        # 计算表格最小所需高度
        min_height = 28 * (len(process_ids) + 1) + 2  # +1 for header, +2 for borders
        min_height = max(min_height, 150)  # 至少150像素
        self.process_table.setMinimumHeight(min_height)
        
        for i, pid in enumerate(process_ids):
            try:
                process = psutil.Process(pid)
                cpu_percent = process.cpu_percent()
                memory_percent = process.memory_percent()
                
                self.process_info[pid] = {
                    'cpu': cpu_percent,
                    'memory': memory_percent
                }
                
                pid_item = QTableWidgetItem(str(pid))
                cpu_item = QTableWidgetItem(f"{cpu_percent:.1f}%")
                mem_item = QTableWidgetItem(f"{memory_percent:.1f}%")
                
                # 确保文本垂直居中
                pid_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                cpu_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                mem_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                self.process_table.setItem(i, 0, pid_item)
                self.process_table.setItem(i, 1, cpu_item)
                self.process_table.setItem(i, 2, mem_item)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.process_table.setItem(i, 0, QTableWidgetItem(str(pid)))
                self.process_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self.process_table.setItem(i, 2, QTableWidgetItem("N/A"))

    def update_progress(self, data):
        """更新进度信息"""
        update_type = data.get('type', '')
        
        if update_type == 'progress_init':
            # 初始化进度信息
            self.total_combinations = data.get('total', 0)
            self.completed_combinations = data.get('completed', 0)
            remaining = data.get('remaining', 0)
            
            self.total_progress_bar.setValue(int(self.completed_combinations / self.total_combinations * 100) if self.total_combinations > 0 else 0)
            self.total_progress_label.setText(f"总进度: {self.completed_combinations}/{self.total_combinations} ({self.completed_combinations / self.total_combinations * 100:.1f}%)")
            
            self.log_message(data.get('message', ''))
            
        elif update_type == 'batch_info':
            # 批次信息
            self.total_batches = data.get('total_batches', 0)
            processes = data.get('processes', 0)
            process_ids = data.get('process_ids', [])  # 获取进程ID列表
            self.active_processes = processes
            self.process_label.setText(f"活跃进程数: {processes}")
            self.update_process_info(process_ids)  # 更新进程信息显示
            self.log_message(data.get('message', ''))
            
        elif update_type == 'process_update':
            # 更新进程信息
            process_ids = data.get('process_ids', [])
            self.update_process_info(process_ids)
            
        elif update_type == 'batch_start':
            # 批次开始
            self.current_batch = data.get('batch_idx', 0) + 1
            self.batch_progress_bar.setValue(0)
            self.batch_progress_label.setText(f"当前批次: {self.current_batch}/{self.total_batches} (0%)")
            self.log_message(data.get('message', ''))
            
        elif update_type == 'batch_progress':
            # 批次进度
            batch_progress = data.get('batch_progress', 0) * 100
            total_progress = data.get('overall_progress', 0) * 100
            
            self.batch_progress_bar.setValue(int(batch_progress))
            self.batch_progress_label.setText(f"当前批次: {self.current_batch}/{self.total_batches} ({batch_progress:.1f}%)")
            
            self.total_progress_bar.setValue(int(total_progress))
            self.total_progress_label.setText(f"总进度: {int(total_progress * self.total_combinations / 100)}/{self.total_combinations} ({total_progress:.1f}%)")
            
            # 更新当前处理的结果
            result = data.get('result', {})
            if 'params' in result:
                self.update_param_display(result['params'])
                
            # 如果有因子结果，添加到表格
            if 'factors' in result:
                for factor in result['factors']:
                    self.add_factor_to_table(factor, result['params'])
            
        elif update_type == 'batch_end':
            # 批次结束
            self.log_message(data.get('message', ''))
            
        elif update_type == 'param_start':
            # 参数开始
            params = data.get('params', {})
            self.update_param_display(params)
            self.log_message(f"开始处理参数组合: forward_period={params.get('forward_period')}, generations={params.get('generations')}, population_size={params.get('population_size')}")
            
        elif update_type == 'param_end':
            # 参数结束
            result = data.get('result', {})
            if 'factors' in result:
                factors = result['factors']
                params = result['params']
                self.log_message(f"参数组合处理完成，找到 {len(factors)} 个因子")
                for factor in factors:
                    self.add_factor_to_table(factor, params)
            
        elif update_type == 'error':
            # 错误信息
            self.log_message(f"错误: {data.get('message', '')}")
            
        elif update_type == 'complete':
            # 完成信息
            self.log_message(data.get('message', ''))
            
        elif update_type == 'stopped':
            # 停止信息
            self.log_message(data.get('message', ''))
            
    def update_param_display(self, params):
        """更新参数显示"""
        for param_name, label in self.param_value_labels.items():
            if param_name in params:
                value_str = str(params[param_name])
                label.setText(value_str)
                
                # 确保值完整显示
                label.setMinimumWidth(label.fontMetrics().horizontalAdvance(value_str) + 20)
                label.setToolTip(value_str)  # 添加工具提示，方便查看完整值
            else:
                label.setText("-")
                label.setToolTip("")
                
    def add_factor_to_table(self, factor, params):
        """添加因子到结果表格"""
        if 'expression' not in factor:
            return
            
        # 避免重复添加相同因子
        factor_key = (params.get('forward_period', 0), factor.get('expression', ''))
        for existing_factor in self.factors_found:
            if (existing_factor[0], existing_factor[1]) == factor_key:
                return
                
        # 将因子添加到列表
        self.factors_found.append((
            params.get('forward_period', 0),
            factor.get('expression', ''),
            factor.get('ic', 0),
            factor.get('stability', 0),
            factor.get('long_returns', 0),
            factor.get('short_returns', 0)
        ))
        
        # 按IC值排序
        self.factors_found.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 更新表格
        self.results_table.setRowCount(len(self.factors_found))
        
        for i, (forward_period, expression, ic, stability, long_returns, short_returns) in enumerate(self.factors_found):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(forward_period)))
            self.results_table.setItem(i, 1, QTableWidgetItem(expression))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{ic:.4f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{stability:.4f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{long_returns:.4f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{short_returns:.4f}"))
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        self.status_log_text.append(log_entry)  # 同时更新状态日志

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格，更接近Windows 11
    ui = GridSearchUI()
    sys.exit(app.exec()) 