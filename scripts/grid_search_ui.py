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
                           QMessageBox, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QInputDialog,
                           QDialog, QVBoxLayout, QTextBrowser)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette

# 导入项目模块
from scripts.grid_search_factors import FactorGridSearch
from factor_research.config.grid_search_config import PARAM_GRID, SPECIAL_COMBINATIONS, FIXED_PARAMS

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
        self.all_processes = []  # 存储所有创建的进程对象，便于强制终止
        
        # 可自定义参数
        self.custom_grid_params = None
        self.custom_fixed_params = None
        self.custom_factor_settings = None
        
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
            
            # 使用自定义的网格参数(如果有)
            param_grid = self.custom_grid_params if self.custom_grid_params else PARAM_GRID
            fixed_params = self.custom_fixed_params if self.custom_fixed_params else FIXED_PARAMS
            
            # 从配置中生成网格参数组合
            forward_periods = param_grid.get("forward_period", [12])
            populations = param_grid.get("population_size", [1000])
            generations = param_grid.get("generations", [100])
            tournament_sizes = param_grid.get("tournament_size", [20])
            
            # 生成所有组合
            for forward_period in forward_periods:
                for population_size in populations:
                    for generation in generations:
                        for tournament_size in tournament_sizes:
                            # 合并固定参数和可变参数
                            params = {
                                "forward_period": forward_period,
                                "population_size": population_size,
                                "generations": generation,
                                "tournament_size": tournament_size,
                                **fixed_params
                            }
                            
                            # 如果有自定义的因子条件，也添加进去
                            if self.custom_factor_settings:
                                # 添加因子条件参数
                                params.update({
                                    "ic_threshold": self.custom_factor_settings.get("ic_threshold", 0.05),
                                    "stability_threshold": self.custom_factor_settings.get("stability_threshold", 0.3),
                                    "min_long_return": self.custom_factor_settings.get("min_long_return", 0.5),
                                    "min_short_return": self.custom_factor_settings.get("min_short_return", -0.5),
                                    "enable_segment_test": self.custom_factor_settings.get("enable_segment_test", True),
                                    "test_set_ratio": self.custom_factor_settings.get("test_set_ratio", 0.3),
                                    "max_complexity": self.custom_factor_settings.get("max_complexity", 20)
                                })
                            
                            all_param_combinations.append(params)
            
            # 添加特殊组合 (如果用户没有指定自定义参数的情况下)
            if not self.custom_grid_params:
                for special_combo in SPECIAL_COMBINATIONS:
                    # 确保特殊组合中包含所有必要的固定参数
                    combo = {**fixed_params, **special_combo}
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
                # 检查是否应该停止 - 在每个批次开始前检查
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
                
                # 使用多进程处理参数组合，增加超时和异常处理
                try:
                    with mp.Pool(processes=num_processes) as pool:
                        self.current_process_pool = pool
                        
                        # 获取进程池中的所有进程ID
                        self.active_process_ids = {p.pid for p in pool._pool}
                        self.update_signal.emit({
                            'type': 'process_update',
                            'process_ids': list(self.active_process_ids)
                        })
                        
                        # 增强版imap处理，添加超时检查
                        result_iter = pool.imap(process_param_search, batch_args)
                        for i in range(len(batch_args)):
                            # 频繁检查是否应该停止
                            if not self.running:
                                pool.terminate()  # 立即终止所有进程
                                pool.join(0.5)    # 短暂等待进程终止
                                self.update_signal.emit({
                                    'type': 'stopped',
                                    'message': "搜索已停止"
                                })
                                return
                                
                            # 检查是否暂停
                            while self.paused and self.running:
                                time.sleep(0.5)
                            
                            # 使用带超时的方式获取结果，避免阻塞
                            try:
                                # 获取下一个结果，设置超时
                                result = None
                                timeout_counter = 0
                                while timeout_counter < 30 and self.running:  # 最多等待30秒
                                    try:
                                        result = result_iter.__next__(timeout=1)  # 每秒检查一次
                                        break
                                    except mp.TimeoutError:
                                        timeout_counter += 1
                                        # 检查是否停止
                                        if not self.running:
                                            pool.terminate()
                                            self.update_signal.emit({
                                                'type': 'stopped',
                                                'message': "搜索已停止"
                                            })
                                            return
                                
                                # 处理结果
                                if result is not None:  # 只添加有效结果
                                    batch_results.append(result)
                                    results.append(result)
                                    
                                    # 实时保存结果
                                    with open(self.searcher.results_file, 'wb') as f:
                                        pickle.dump(results, f)
                            except StopIteration:
                                # 所有结果都已处理完
                                break
                            except Exception as e:
                                self.log_message(f"处理结果时出错: {str(e)}")
                                continue
                            
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
                
                except Exception as e:
                    self.update_signal.emit({
                        'type': 'error',
                        'message': f"处理批次时出错: {str(e)}"
                    })
                    if not self.running:
                        return
                
                # 如果不再运行，退出循环
                if not self.running:
                    return
                    
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
            if self.running:
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
                try:
                    self.current_process_pool.terminate()
                    self.current_process_pool.join(1)  # 等待最多1秒
                    self.current_process_pool.close()
                except:
                    pass
            
            # 确保所有子进程都已终止
            for pid in list(self.active_process_ids):
                try:
                    import psutil
                    process = psutil.Process(pid)
                    process.terminate()
                except:
                    pass
            
            # 清空进程ID集合
            self.active_process_ids.clear()
            
            import gc
            gc.collect()
            self.finished_signal.emit()
        
    def stop(self):
        """停止搜索"""
        # 更新状态信号
        self.update_signal.emit({
            'type': 'stopping',
            'message': "正在终止所有进程，请稍候..."
        })
        
        self.running = False
        
        # 1. 如果有正在运行的进程池，立即终止它
        if self.current_process_pool is not None:
            try:
                self.current_process_pool.terminate()
                self.current_process_pool.join(1)  # 等待最多1秒钟
                self.current_process_pool.close()
            except Exception as e:
                print(f"终止进程池出错: {str(e)}")
        
        # 2. 强制终止所有子进程
        for pid in list(self.active_process_ids):
            try:
                import psutil
                process = psutil.Process(pid)
                # 终止进程及其子进程
                for child in process.children(recursive=True):
                    try:
                        child.terminate()
                    except:
                        try:
                            child.kill()
                        except:
                            pass
                # 终止主进程
                process.terminate()
            except Exception as e:
                print(f"终止进程 {pid} 出错: {str(e)}")
        
        # 3. 清空进程ID集合
        self.active_process_ids.clear()
        
        # 4. 显示停止消息
        self.update_signal.emit({
            'type': 'stopped',
            'message': "搜索已停止"
        })
        
        # 5. 发送完成信号
        self.finished_signal.emit()
    
    def pause(self):
        """暂停搜索"""
        self.paused = True
        
    def resume(self):
        """恢复搜索"""
        self.paused = False

class GridSearchUI(QWidget):
    """因子网格搜索UI"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化变量
        self.worker = None
        self.process_info = {}
        self.active_processes = 0
        self.factors_found = []
        
        # 添加这些缺失的变量
        self.current_batch = 0
        self.total_batches = 0
        self.total_combinations = 0
        self.completed_combinations = 0
        
        # 初始化因子条件和网格参数的默认值
        self.factor_settings = {
            "ic_threshold": 0.05,
            "stability_threshold": 0.3,
            "min_long_return": 0.5,
            "min_short_return": -0.5,
            "enable_segment_test": True,
            "test_set_ratio": 0.3,
            "max_complexity": 20,
            "transaction_fee": 0.1,    # 添加交易手续费默认值 0.1%
            "min_trade_return": 0.3    # 添加单次交易最小收益默认值 0.3%
        }
        
        self.grid_params = {
            "forward_period": [12],
            "population_size": [1000],
            "generations": [100],
            "tournament_size": [20]
        }
        
        self.fixed_params = {
            "p_crossover": 0.5,
            "p_subtree_mutation": 0.2,
            "p_hoist_mutation": 0.1, 
            "p_point_mutation": 0.1,
            "parsimony_coefficient": 0.001
        }
        
        # 设置内存监控定时器
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.setInterval(2000)  # 2秒更新一次内存使用率
        
        # 设置进程监控定时器
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.update_process_resources)
        self.process_timer.setInterval(2000)  # 2秒更新一次进程资源使用率
        
        # 初始化UI
        self.init_ui()
        
        # 记录日志
        self.log_message("系统已启动，等待开始网格搜索...")
        
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
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
        
        # ======= 5. 因子条件设置标签页 =======
        factor_settings_tab = QWidget()
        factor_settings_layout = QVBoxLayout(factor_settings_tab)
        factor_settings_layout.setContentsMargins(15, 15, 15, 15)
        factor_settings_layout.setSpacing(15)
        
        # 创建因子条件设置组
        factor_conditions_group = QGroupBox("因子条件设置")
        factor_conditions_layout = QGridLayout(factor_conditions_group)
        factor_conditions_layout.setContentsMargins(15, 25, 15, 15)
        factor_conditions_layout.setSpacing(20)
        
        # IC值最小阈值设置
        row = 0
        factor_conditions_layout.addWidget(QLabel("IC值最小阈值:"), row, 0)
        self.ic_threshold_spinbox = QDoubleSpinBox()
        self.ic_threshold_spinbox.setRange(0.0, 1.0)
        self.ic_threshold_spinbox.setSingleStep(0.01)
        self.ic_threshold_spinbox.setValue(0.05)  # 默认值
        self.ic_threshold_spinbox.setToolTip("IC值绝对值低于此阈值的因子将被过滤掉")
        
        # 添加单位和帮助按钮的布局
        ic_layout = QHBoxLayout()
        ic_layout.addWidget(self.ic_threshold_spinbox)
        ic_layout.addWidget(QLabel("绝对值"))  # 添加单位标识
        
        # 添加帮助按钮
        ic_help_btn = QPushButton("帮助")
        ic_help_btn.setMinimumWidth(60)
        ic_help_btn.setFixedHeight(28)
        ic_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        ic_help_btn.clicked.connect(lambda: self.show_param_help("IC值最小阈值", 
            "IC值(信息系数)是衡量因子预测能力的重要指标，范围在-1到1之间。\n\n"
            "- 绝对值越高：表示因子的预测能力越强\n"
            "- 正值：表示因子与未来收益正相关\n"
            "- 负值：表示因子与未来收益负相关\n\n"
            "调高此阈值会筛选出预测能力更强的因子，但可能减少找到的因子数量。\n"
            "一般建议设置在0.05-0.2之间。"
        ))
        ic_layout.addWidget(ic_help_btn)
        ic_layout.addStretch()
        
        factor_conditions_layout.addLayout(ic_layout, row, 1)
        
        # 稳定性最小阈值设置
        row += 1
        factor_conditions_layout.addWidget(QLabel("稳定性最小阈值:"), row, 0)
        self.stability_threshold_spinbox = QDoubleSpinBox()
        self.stability_threshold_spinbox.setRange(0.0, 1.0)
        self.stability_threshold_spinbox.setSingleStep(0.01)
        self.stability_threshold_spinbox.setValue(0.3)  # 默认值
        self.stability_threshold_spinbox.setToolTip("稳定性低于此阈值的因子将被过滤掉")
        
        # 添加单位和帮助按钮
        stability_layout = QHBoxLayout()
        stability_layout.addWidget(self.stability_threshold_spinbox)
        stability_layout.addWidget(QLabel("比率"))  # 添加单位标识
        
        # 添加帮助按钮
        stability_help_btn = QPushButton("帮助")
        stability_help_btn.setMinimumWidth(60)
        stability_help_btn.setFixedHeight(28)
        stability_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        stability_help_btn.clicked.connect(lambda: self.show_param_help("稳定性最小阈值", 
            "稳定性衡量因子在不同时间段内表现的一致性，范围在0到1之间。\n\n"
            "- 值越高：表示因子在不同市场环境下表现越稳定\n"
            "- 值越低：表示因子的表现波动较大，可能仅在特定市场环境下有效\n\n"
            "调高此阈值会筛选出更稳定的因子，减少过拟合风险，但可能减少找到的因子数量。\n"
            "一般建议设置在0.3-0.6之间。"
        ))
        stability_layout.addWidget(stability_help_btn)
        stability_layout.addStretch()
        
        factor_conditions_layout.addLayout(stability_layout, row, 1)
        
        # 最小做多收益要求
        row += 1
        factor_conditions_layout.addWidget(QLabel("最小做多收益要求:"), row, 0)
        self.min_long_return_spinbox = QDoubleSpinBox()
        self.min_long_return_spinbox.setRange(-1.0, 10.0)
        self.min_long_return_spinbox.setSingleStep(0.1)
        self.min_long_return_spinbox.setValue(0.5)  # 默认值
        self.min_long_return_spinbox.setToolTip("做多收益低于此阈值的因子将被过滤掉")
        
        # 添加单位和帮助按钮
        long_return_layout = QHBoxLayout()
        long_return_layout.addWidget(self.min_long_return_spinbox)
        long_return_layout.addWidget(QLabel("%/天"))  # 添加单位标识
        
        # 添加帮助按钮
        long_help_btn = QPushButton("帮助")
        long_help_btn.setMinimumWidth(60)
        long_help_btn.setFixedHeight(28)
        long_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        long_help_btn.clicked.connect(lambda: self.show_param_help("最小做多收益要求", 
            "最小做多收益指根据因子选择标的做多策略的最低期望收益率，单位为%/天。\n\n"
            "- 值越高：要求因子有更强的做多选股能力\n"
            "- 值越低：放宽对因子做多能力的要求\n\n"
            "调高此阈值会筛选出做多效果更好的因子，但可能减少找到的因子数量。\n"
            "一般建议设置在0.1%-1.0%/天之间。\n\n"
            "例如：0.5表示预期每天至少有0.5%的收益率。"
        ))
        long_return_layout.addWidget(long_help_btn)
        long_return_layout.addStretch()
        
        factor_conditions_layout.addLayout(long_return_layout, row, 1)
        
        # 最小做空收益要求
        row += 1
        factor_conditions_layout.addWidget(QLabel("最小做空收益要求:"), row, 0)
        self.min_short_return_spinbox = QDoubleSpinBox()
        self.min_short_return_spinbox.setRange(-10.0, 1.0)
        self.min_short_return_spinbox.setSingleStep(0.1)
        self.min_short_return_spinbox.setValue(-0.5)  # 默认值
        self.min_short_return_spinbox.setToolTip("做空收益高于此阈值的因子将被过滤掉")
        
        # 添加单位和帮助按钮
        short_return_layout = QHBoxLayout()
        short_return_layout.addWidget(self.min_short_return_spinbox)
        short_return_layout.addWidget(QLabel("%/天"))  # 添加单位标识
        
        # 添加帮助按钮
        short_help_btn = QPushButton("帮助")
        short_help_btn.setMinimumWidth(60)
        short_help_btn.setFixedHeight(28)
        short_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        short_help_btn.clicked.connect(lambda: self.show_param_help("最小做空收益要求", 
            "最小做空收益指根据因子选择标的做空策略的最低期望收益率，单位为%/天。\n\n"
            "- 值越低（负值更大）：要求因子有更强的做空选股能力\n"
            "- 值越高（接近0）：放宽对因子做空能力的要求\n\n"
            "注意：做空收益通常为负值，因此此处设置的是上限而非下限。\n"
            "调低此阈值会筛选出做空效果更好的因子，但可能减少找到的因子数量。\n"
            "一般建议设置在-1.0%到-0.1%/天之间。\n\n"
            "例如：-0.5表示预期每天至少有0.5%的负收益率（做空获利）。"
        ))
        short_return_layout.addWidget(short_help_btn)
        short_return_layout.addStretch()
        
        factor_conditions_layout.addLayout(short_return_layout, row, 1)
        
        # 启用分段测试
        row += 1
        factor_conditions_layout.addWidget(QLabel("启用分段测试:"), row, 0)
        segment_layout = QHBoxLayout()
        
        self.enable_segment_test_checkbox = QCheckBox("将数据分为训练集和测试集")
        self.enable_segment_test_checkbox.setChecked(True)
        self.enable_segment_test_checkbox.setToolTip("启用后将数据分为训练集和测试集，更好地评估因子的泛化能力")
        segment_layout.addWidget(self.enable_segment_test_checkbox)
        
        # 添加帮助按钮
        segment_help_btn = QPushButton("帮助")
        segment_help_btn.setMinimumWidth(60)
        segment_help_btn.setFixedHeight(28)
        segment_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        segment_help_btn.clicked.connect(lambda: self.show_param_help("启用分段测试", 
            "分段测试是评估因子泛化能力的重要方法，通过将数据分为训练集和测试集。\n\n"
            "- 启用：因子首先在训练集上寻找，然后在测试集上验证，能够更好地防止过拟合\n"
            "- 禁用：使用全部数据寻找因子，可能导致因子过拟合历史数据\n\n"
            "建议保持启用状态，以获得具有更好泛化能力的因子。\n"
            "只有在数据量极其有限的情况下，才考虑禁用此功能。"
        ))
        segment_layout.addWidget(segment_help_btn)
        segment_layout.addStretch()
        
        factor_conditions_layout.addLayout(segment_layout, row, 1)
        
        # 测试集比例
        row += 1
        factor_conditions_layout.addWidget(QLabel("测试集比例:"), row, 0)
        self.test_set_ratio_spinbox = QDoubleSpinBox()
        self.test_set_ratio_spinbox.setRange(0.1, 0.5)
        self.test_set_ratio_spinbox.setSingleStep(0.05)
        self.test_set_ratio_spinbox.setValue(0.3)  # 默认值
        self.test_set_ratio_spinbox.setToolTip("测试集占总数据的比例")
        
        # 添加单位和帮助按钮
        test_ratio_layout = QHBoxLayout()
        test_ratio_layout.addWidget(self.test_set_ratio_spinbox)
        test_ratio_layout.addWidget(QLabel("比例"))  # 添加单位标识
        
        # 添加帮助按钮
        test_ratio_help_btn = QPushButton("帮助")
        test_ratio_help_btn.setMinimumWidth(60)
        test_ratio_help_btn.setFixedHeight(28)
        test_ratio_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        test_ratio_help_btn.clicked.connect(lambda: self.show_param_help("测试集比例", 
            "测试集比例决定了数据集中多大比例用于因子验证，而非因子寻找。\n\n"
            "- 值越高：验证更严格，但可用于寻找因子的数据减少\n"
            "- 值越低：有更多数据用于寻找因子，但验证可能不够充分\n\n"
            "调高此值会提高因子的可靠性，但可能减少找到的因子数量。\n"
            "一般推荐设置在0.2-0.4之间。\n\n"
            "例如：0.3表示30%的数据用于测试，70%用于训练。"
        ))
        test_ratio_layout.addWidget(test_ratio_help_btn)
        test_ratio_layout.addStretch()
        
        factor_conditions_layout.addLayout(test_ratio_layout, row, 1)
        
        # 最大因子复杂度
        row += 1
        factor_conditions_layout.addWidget(QLabel("最大因子复杂度:"), row, 0)
        self.max_complexity_spinbox = QSpinBox()
        self.max_complexity_spinbox.setRange(5, 50)
        self.max_complexity_spinbox.setSingleStep(1)
        self.max_complexity_spinbox.setValue(20)  # 默认值
        self.max_complexity_spinbox.setToolTip("控制生成因子的最大复杂度，影响因子的可解释性")
        
        # 添加单位和帮助按钮
        complexity_layout = QHBoxLayout()
        complexity_layout.addWidget(self.max_complexity_spinbox)
        complexity_layout.addWidget(QLabel("节点数"))  # 添加单位标识
        
        # 添加帮助按钮
        complexity_help_btn = QPushButton("帮助")
        complexity_help_btn.setMinimumWidth(60)
        complexity_help_btn.setFixedHeight(28)
        complexity_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        complexity_help_btn.clicked.connect(lambda: self.show_param_help("最大因子复杂度", 
            "因子复杂度指遗传编程生成的表达式树的大小，以节点数计算。\n\n"
            "- 值越高：允许生成更复杂的因子，可能具有更高的拟合能力，但可解释性降低\n"
            "- 值越低：限制因子复杂度，提高可解释性，但可能限制因子表达能力\n\n"
            "调高此值会增加寻找到更强因子的可能性，但降低可解释性，并可能导致过拟合。\n"
            "一般建议设置在10-30节点之间。\n\n"
            "例如：20表示因子表达式最多包含20个操作和变量节点。"
        ))
        complexity_layout.addWidget(complexity_help_btn)
        complexity_layout.addStretch()
        
        factor_conditions_layout.addLayout(complexity_layout, row, 1)
        
        # 交易手续费设置
        row += 1
        factor_conditions_layout.addWidget(QLabel("交易手续费率:"), row, 0)
        self.transaction_fee_spinbox = QDoubleSpinBox()
        self.transaction_fee_spinbox.setRange(0.0, 1.0)
        self.transaction_fee_spinbox.setSingleStep(0.01)
        self.transaction_fee_spinbox.setDecimals(3)
        self.transaction_fee_spinbox.setValue(0.1)  # 默认值 0.1%
        self.transaction_fee_spinbox.setToolTip("设置交易手续费率，用于因子评估")
        
        # 添加单位和帮助按钮
        fee_layout = QHBoxLayout()
        fee_layout.addWidget(self.transaction_fee_spinbox)
        fee_layout.addWidget(QLabel("%/笔"))  # 添加单位标识
        
        # 添加帮助按钮
        fee_help_btn = QPushButton("帮助")
        fee_help_btn.setMinimumWidth(60)
        fee_help_btn.setFixedHeight(28)
        fee_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        fee_help_btn.clicked.connect(lambda: self.show_param_help("交易手续费率", 
            "交易手续费率是评估因子实际表现的关键参数，影响因子产生的净收益。\n\n"
            "- 设置过低：可能导致过度乐观的因子评估结果\n"
            "- 设置过高：可能过度保守，错误地排除有效因子\n\n"
            "建议设置为您通常交易环境中的实际费率，通常在0.05%-0.3%之间。\n"
            "这将使因子评估更接近实际交易环境，筛选出在考虑成本后依然有效的因子。\n\n"
            "例如：0.1%表示每笔交易收取总交易金额的0.1%作为手续费。"
        ))
        fee_layout.addWidget(fee_help_btn)
        fee_layout.addStretch()
        
        factor_conditions_layout.addLayout(fee_layout, row, 1)
        
        # 单次交易最小收益要求
        row += 1
        factor_conditions_layout.addWidget(QLabel("单次交易最小收益:"), row, 0)
        self.min_trade_return_spinbox = QDoubleSpinBox()
        self.min_trade_return_spinbox.setRange(0.0, 5.0)
        self.min_trade_return_spinbox.setSingleStep(0.05)
        self.min_trade_return_spinbox.setDecimals(2)
        self.min_trade_return_spinbox.setValue(0.3)  # 默认值 0.3%
        self.min_trade_return_spinbox.setToolTip("设置单次交易需要达到的最小收益率")
        
        # 添加单位和帮助按钮
        trade_return_layout = QHBoxLayout()
        trade_return_layout.addWidget(self.min_trade_return_spinbox)
        trade_return_layout.addWidget(QLabel("%/笔"))  # 添加单位标识
        
        # 添加帮助按钮
        trade_return_help_btn = QPushButton("帮助")
        trade_return_help_btn.setMinimumWidth(60)
        trade_return_help_btn.setFixedHeight(28)
        trade_return_help_btn.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;")
        trade_return_help_btn.clicked.connect(lambda: self.show_param_help("单次交易最小收益", 
            "单次交易最小收益设置每笔交易在扣除手续费后需要达到的最低净收益率。\n\n"
            "- 值越高：筛选出信号更强、收益更显著的交易机会\n"
            "- 值越低：允许更多小额收益交易，可能增加交易频率\n\n"
            "此参数有助于过滤掉微小收益的频繁交易，减少因过度交易导致的成本侵蚀。\n"
            "建议设置为至少大于交易手续费的2-3倍，确保每笔交易都有足够的利润空间。\n\n"
            "例如：设置0.3%意味着只有预期收益超过0.3%的交易信号才会被执行。"
        ))
        trade_return_layout.addWidget(trade_return_help_btn)
        trade_return_layout.addStretch()
        
        factor_conditions_layout.addLayout(trade_return_layout, row, 1)
        
        # 应用设置按钮
        row += 1
        self.apply_factor_settings_button = QPushButton("应用因子条件设置")
        self.apply_factor_settings_button.clicked.connect(self.apply_factor_settings)
        factor_conditions_layout.addWidget(self.apply_factor_settings_button, row, 0, 1, 2)
        
        factor_settings_layout.addWidget(factor_conditions_group)
        main_tabs.addTab(factor_settings_tab, "因子条件设置")
        
        # ======= 6. 网格参数设置标签页 =======
        grid_params_tab = QWidget()
        grid_params_layout = QVBoxLayout(grid_params_tab)
        grid_params_layout.setContentsMargins(15, 15, 15, 15)
        grid_params_layout.setSpacing(15)
        
        # 创建网格参数设置组
        grid_params_group = QGroupBox("网格参数设置")
        grid_params_grid = QGridLayout(grid_params_group)
        grid_params_grid.setContentsMargins(15, 25, 15, 15)
        grid_params_grid.setSpacing(20)
        
        # 预测周期设置
        row = 0
        grid_params_grid.addWidget(QLabel("预测周期:"), row, 0)
        self.forward_period_layout = QHBoxLayout()
        
        # 预测周期-添加按钮
        self.forward_period_add_button = QPushButton("+")
        self.forward_period_add_button.setMaximumWidth(30)
        self.forward_period_add_button.clicked.connect(lambda: self.add_period_value("forward_period"))
        
        # 预测周期-删除按钮
        self.forward_period_remove_button = QPushButton("-")
        self.forward_period_remove_button.setMaximumWidth(30)
        self.forward_period_remove_button.clicked.connect(lambda: self.remove_param_value("forward_period"))
        
        # 预测周期-已选列表
        self.forward_period_label = QLabel("已选: 12")
        
        self.forward_period_layout.addWidget(self.forward_period_add_button)
        self.forward_period_layout.addWidget(self.forward_period_remove_button)
        self.forward_period_layout.addWidget(self.forward_period_label)
        self.forward_period_layout.addStretch()
        
        grid_params_grid.addLayout(self.forward_period_layout, row, 1)
        
        # 存储预测周期值
        self.forward_periods = [12]  # 默认值
        
        # 种群大小设置
        row += 1
        grid_params_grid.addWidget(QLabel("种群大小:"), row, 0)
        self.population_size_layout = QHBoxLayout()
        
        # 种群大小-添加按钮
        self.population_size_add_button = QPushButton("+")
        self.population_size_add_button.setMaximumWidth(30)
        self.population_size_add_button.clicked.connect(lambda: self.add_int_value("population_size"))
        
        # 种群大小-删除按钮
        self.population_size_remove_button = QPushButton("-")
        self.population_size_remove_button.setMaximumWidth(30)
        self.population_size_remove_button.clicked.connect(lambda: self.remove_param_value("population_size"))
        
        # 种群大小-已选列表
        self.population_size_label = QLabel("已选: 1000")
        
        self.population_size_layout.addWidget(self.population_size_add_button)
        self.population_size_layout.addWidget(self.population_size_remove_button)
        self.population_size_layout.addWidget(self.population_size_label)
        self.population_size_layout.addStretch()
        
        grid_params_grid.addLayout(self.population_size_layout, row, 1)
        
        # 存储种群大小值
        self.population_sizes = [1000]  # 默认值
        
        # 进化代数设置
        row += 1
        grid_params_grid.addWidget(QLabel("进化代数:"), row, 0)
        self.generations_layout = QHBoxLayout()
        
        # 进化代数-添加按钮
        self.generations_add_button = QPushButton("+")
        self.generations_add_button.setMaximumWidth(30)
        self.generations_add_button.clicked.connect(lambda: self.add_int_value("generations"))
        
        # 进化代数-删除按钮
        self.generations_remove_button = QPushButton("-")
        self.generations_remove_button.setMaximumWidth(30)
        self.generations_remove_button.clicked.connect(lambda: self.remove_param_value("generations"))
        
        # 进化代数-已选列表
        self.generations_label = QLabel("已选: 100")
        
        self.generations_layout.addWidget(self.generations_add_button)
        self.generations_layout.addWidget(self.generations_remove_button)
        self.generations_layout.addWidget(self.generations_label)
        self.generations_layout.addStretch()
        
        grid_params_grid.addLayout(self.generations_layout, row, 1)
        
        # 存储进化代数值
        self.generations_values = [100]  # 默认值
        
        # 锦标赛大小设置
        row += 1
        grid_params_grid.addWidget(QLabel("锦标赛大小:"), row, 0)
        self.tournament_size_layout = QHBoxLayout()
        
        # 锦标赛大小-添加按钮
        self.tournament_size_add_button = QPushButton("+")
        self.tournament_size_add_button.setMaximumWidth(30)
        self.tournament_size_add_button.clicked.connect(lambda: self.add_int_value("tournament_size"))
        
        # 锦标赛大小-删除按钮
        self.tournament_size_remove_button = QPushButton("-")
        self.tournament_size_remove_button.setMaximumWidth(30)
        self.tournament_size_remove_button.clicked.connect(lambda: self.remove_param_value("tournament_size"))
        
        # 锦标赛大小-已选列表
        self.tournament_size_label = QLabel("已选: 20")
        
        self.tournament_size_layout.addWidget(self.tournament_size_add_button)
        self.tournament_size_layout.addWidget(self.tournament_size_remove_button)
        self.tournament_size_layout.addWidget(self.tournament_size_label)
        self.tournament_size_layout.addStretch()
        
        grid_params_grid.addLayout(self.tournament_size_layout, row, 1)
        
        # 存储锦标赛大小值
        self.tournament_sizes = [20]  # 默认值
        
        # 交叉概率设置
        row += 1
        grid_params_grid.addWidget(QLabel("交叉概率:"), row, 0)
        self.p_crossover_spinbox = QDoubleSpinBox()
        self.p_crossover_spinbox.setRange(0.1, 0.6)
        self.p_crossover_spinbox.setSingleStep(0.05)
        self.p_crossover_spinbox.setValue(0.5)  # 默认值改为0.5
        self.p_crossover_spinbox.setToolTip("控制遗传算法中交叉操作的概率，与其他变异概率之和不能超过1.0")
        grid_params_grid.addWidget(self.p_crossover_spinbox, row, 1)
        
        # 子树变异概率设置
        row += 1
        grid_params_grid.addWidget(QLabel("子树变异概率:"), row, 0)
        self.p_subtree_mutation_spinbox = QDoubleSpinBox()
        self.p_subtree_mutation_spinbox.setRange(0.0, 0.3)
        self.p_subtree_mutation_spinbox.setSingleStep(0.05)
        self.p_subtree_mutation_spinbox.setValue(0.2)  # 默认值
        self.p_subtree_mutation_spinbox.setToolTip("控制子树变异操作的概率，所有变异概率与交叉概率之和需小于或等于1.0")
        grid_params_grid.addWidget(self.p_subtree_mutation_spinbox, row, 1)
        
        # 提升变异概率设置
        row += 1
        grid_params_grid.addWidget(QLabel("提升变异概率:"), row, 0)
        self.p_hoist_mutation_spinbox = QDoubleSpinBox()
        self.p_hoist_mutation_spinbox.setRange(0.0, 0.3)
        self.p_hoist_mutation_spinbox.setSingleStep(0.05)
        self.p_hoist_mutation_spinbox.setValue(0.1)  # 默认值
        self.p_hoist_mutation_spinbox.setToolTip("控制提升变异操作的概率，所有变异概率与交叉概率之和需小于或等于1.0")
        grid_params_grid.addWidget(self.p_hoist_mutation_spinbox, row, 1)
        
        # 点变异概率设置
        row += 1
        grid_params_grid.addWidget(QLabel("点变异概率:"), row, 0)
        self.p_point_mutation_spinbox = QDoubleSpinBox()
        self.p_point_mutation_spinbox.setRange(0.0, 0.3)
        self.p_point_mutation_spinbox.setSingleStep(0.05)
        self.p_point_mutation_spinbox.setValue(0.1)  # 默认值
        self.p_point_mutation_spinbox.setToolTip("控制点变异操作的概率，所有变异概率与交叉概率之和需小于或等于1.0")
        grid_params_grid.addWidget(self.p_point_mutation_spinbox, row, 1)
        
        # 复杂度惩罚系数设置
        row += 1
        grid_params_grid.addWidget(QLabel("复杂度惩罚系数:"), row, 0)
        self.parsimony_coefficient_spinbox = QDoubleSpinBox()
        self.parsimony_coefficient_spinbox.setRange(0.0, 0.1)
        self.parsimony_coefficient_spinbox.setSingleStep(0.001)
        self.parsimony_coefficient_spinbox.setDecimals(4)
        self.parsimony_coefficient_spinbox.setValue(0.001)  # 默认值
        self.parsimony_coefficient_spinbox.setToolTip("控制模型复杂度的惩罚力度，值越大倾向于更简单的模型")
        grid_params_grid.addWidget(self.parsimony_coefficient_spinbox, row, 1)
        
        # 可能的组合数显示
        row += 1
        grid_params_grid.addWidget(QLabel("可能的组合数:"), row, 0)
        self.combinations_label = QLabel("1种组合")
        self.combinations_label.setStyleSheet("font-weight: bold; color: #0078d7;")
        grid_params_grid.addWidget(self.combinations_label, row, 1)
        
        # 应用设置和更新组合数按钮
        row += 1
        self.apply_grid_params_button = QPushButton("应用网格参数设置")
        self.apply_grid_params_button.clicked.connect(self.apply_grid_params)
        grid_params_grid.addWidget(self.apply_grid_params_button, row, 0, 1, 2)
        
        grid_params_layout.addWidget(grid_params_group)
        main_tabs.addTab(grid_params_tab, "网格参数设置")
        
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
        
        # 明确设置按钮的可见性
        self.start_button.setVisible(True)
        self.pause_button.setVisible(True)
        self.stop_button.setVisible(True)
        self.report_button.setVisible(True)
        
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
        
        # 计算组合数并记录日志
        count = (
            len(self.forward_periods) *
            len(self.population_sizes) *
            len(self.generations_values) *
            len(self.tournament_sizes)
        )
        self.log_message(f"将执行{count}种参数组合的搜索")
        
        # 记录当前的因子条件和网格参数
        self.log_message(f"因子条件: IC阈值={self.factor_settings['ic_threshold']}, " +
                       f"稳定性阈值={self.factor_settings['stability_threshold']}, " +
                       f"做多收益要求={self.factor_settings['min_long_return']}, " +
                       f"做空收益要求={self.factor_settings['min_short_return']}")
        
        self.log_message(f"网格参数: 预测周期={self.forward_periods}, " +
                       f"种群大小={self.population_sizes}, " +
                       f"进化代数={self.generations_values}, " +
                       f"锦标赛大小={self.tournament_sizes}")
        
        # 更新按钮状态
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.report_button.setEnabled(False)
        
        # 确保暂停按钮显示正确的文本
        self.pause_button.setText("暂停")
        
        # 创建并启动worker线程，传递用户设置的参数
        self.worker = GridSearchWorker()
        
        # 设置worker的自定义参数
        self.worker.custom_grid_params = self.grid_params
        self.worker.custom_fixed_params = self.fixed_params
        self.worker.custom_factor_settings = self.factor_settings
        
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
            "确定要停止搜索吗？已完成的结果会被保存。\n\n注意：停止过程可能需要几秒钟才能终止所有进程。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 更新UI状态
            self.status_label.setText("状态: 正在停止进程...")
            self.log_message("正在停止搜索，终止所有进程...")
            
            # 禁用停止按钮，避免重复点击
            self.stop_button.setEnabled(False)
            self.stop_button.setText("正在停止...")
            
            # 在UI层面立即停止worker线程
            self.worker.stop()
            
            # 创建一个监控计时器，如果10秒后进程仍未停止，就强制结束
            force_stop_timer = QTimer(self)
            force_stop_timer.setSingleShot(True)
            force_stop_timer.timeout.connect(self.force_stop_all_processes)
            force_stop_timer.start(10000)  # 10秒后触发
    
    def force_stop_all_processes(self):
        """如果常规停止失败，强制停止所有Python进程"""
        if self.worker and self.worker.isRunning():
            self.log_message("正常停止失败，尝试强制终止所有搜索进程...")
            
            try:
                # 获取所有进程中与网格搜索相关的Python进程
                import psutil
                grid_search_processes = []
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # 检查是否是Python进程且命令行中含有grid_search
                        if proc.name() == 'python.exe' or proc.name() == 'python':
                            cmdline = ' '.join(proc.cmdline()).lower()
                            if 'grid_search' in cmdline and proc.pid != os.getpid():
                                grid_search_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                
                # 终止找到的所有进程
                for proc in grid_search_processes:
                    try:
                        self.log_message(f"强制终止进程: {proc.pid}")
                        proc.terminate()
                    except:
                        try:
                            proc.kill()
                        except:
                            pass
                
                # 等待所有进程结束
                psutil.wait_procs(grid_search_processes, timeout=3)
                
                # 更新UI状态
                self.search_finished()
                self.log_message("强制停止完成，所有搜索进程已终止")
                
            except Exception as e:
                self.log_message(f"强制停止过程中出错: {str(e)}")
                # 尽管出错，仍然更新UI状态
                self.search_finished()
    
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
            if result is not None and 'params' in result:
                self.update_param_display(result['params'])
                
            # 如果有因子结果，添加到表格
            if result is not None and 'factors' in result:
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
            if result is not None and 'factors' in result:
                factors = result['factors']
                params = result['params']
                self.log_message(f"参数组合处理完成，找到 {len(factors)} 个因子")
                for factor in factors:
                    self.add_factor_to_table(factor, params)
        
        elif update_type == 'stopping':
            # 正在停止过程中
            self.status_label.setText("状态: 正在终止进程...")
            self.log_message(data.get('message', ''))
            
        elif update_type == 'stopped':
            # 停止完成
            self.status_label.setText("状态: 已停止")
            self.log_message(data.get('message', ''))
            self.search_finished()
            
        elif update_type == 'error':
            # 错误信息
            self.log_message(f"错误: {data.get('message', '')}")
            
        elif update_type == 'complete':
            # 完成信息
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

    def update(self):
        """刷新UI元素显示，确保按钮等元素正确显示"""
        super().update()
        
        # 确保按钮可见
        self.start_button.setVisible(True)
        self.pause_button.setVisible(True)
        self.stop_button.setVisible(True)
        self.report_button.setVisible(True)
        
        # 更新按钮状态
        if self.worker and self.worker.isRunning():
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            if self.worker.paused:
                self.pause_button.setText("继续")
            else:
                self.pause_button.setText("暂停")
        else:
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.pause_button.setText("暂停")
        
        # 强制刷新布局
        self.layout().update()
        
        # 更新进程资源信息
        self.update_memory_usage()
        if self.process_info:
            self.update_process_resources()

    def add_period_value(self, param_type):
        """添加预测周期值"""
        # 根据不同参数类型显示不同的默认值和步长
        if param_type == "forward_period":
            default_value = 12
            values_list = self.forward_periods
            label = self.forward_period_label
            title = "添加预测周期"
            desc = "输入预测周期值 (1-48):"
            min_val = 1
            max_val = 48
        
        value, ok = QInputDialog.getInt(
            self, title, desc, default_value, min_val, max_val, 1
        )
        
        if ok:
            # 检查值是否已存在
            if value not in values_list:
                values_list.append(value)
                # 更新标签显示
                values_str = ", ".join(map(str, sorted(values_list)))
                label.setText(f"已选: {values_str}")
                # 更新组合数
                self.update_combinations_count()
    
    def add_int_value(self, param_type):
        """添加整数值参数"""
        # 根据不同参数类型显示不同的默认值和步长
        if param_type == "population_size":
            default_value = 1000
            values_list = self.population_sizes
            label = self.population_size_label
            title = "添加种群大小"
            desc = "输入种群大小 (100-5000):"
            min_val = 100
            max_val = 5000
            step = 100
        elif param_type == "generations":
            default_value = 100
            values_list = self.generations_values
            label = self.generations_label
            title = "添加进化代数"
            desc = "输入进化代数 (50-500):"
            min_val = 50
            max_val = 500
            step = 10
        elif param_type == "tournament_size":
            default_value = 20
            values_list = self.tournament_sizes
            label = self.tournament_size_label
            title = "添加锦标赛大小"
            desc = "输入锦标赛大小 (5-100):"
            min_val = 5
            max_val = 100
            step = 5
        else:
            return
        
        value, ok = QInputDialog.getInt(
            self, title, desc, default_value, min_val, max_val, step
        )
        
        if ok:
            # 检查值是否已存在
            if value not in values_list:
                values_list.append(value)
                # 更新标签显示
                values_str = ", ".join(map(str, sorted(values_list)))
                label.setText(f"已选: {values_str}")
                # 更新组合数
                self.update_combinations_count()
    
    def update_combinations_count(self):
        """更新可能的组合数显示"""
        # 计算所有可能的组合数
        count = (
            len(self.forward_periods) *
            len(self.population_sizes) *
            len(self.generations_values) *
            len(self.tournament_sizes)
        )
        
        # 更新显示
        self.combinations_label.setText(f"{count}种组合")
        
        # 如果组合数过大，显示警告
        if count > 100:
            self.combinations_label.setStyleSheet("font-weight: bold; color: red;")
        else:
            self.combinations_label.setStyleSheet("font-weight: bold; color: #0078d7;")
    
    def apply_factor_settings(self):
        """应用因子条件设置"""
        # 获取设置值
        ic_threshold = self.ic_threshold_spinbox.value()
        stability_threshold = self.stability_threshold_spinbox.value()
        min_long_return = self.min_long_return_spinbox.value()
        min_short_return = self.min_short_return_spinbox.value()
        enable_segment_test = self.enable_segment_test_checkbox.isChecked()
        test_set_ratio = self.test_set_ratio_spinbox.value()
        max_complexity = self.max_complexity_spinbox.value()
        transaction_fee = self.transaction_fee_spinbox.value()
        min_trade_return = self.min_trade_return_spinbox.value()
        
        # 将设置保存到配置中
        # 实际应用中应该更新到FIXED_PARAMS中
        self.factor_settings = {
            "ic_threshold": ic_threshold,
            "stability_threshold": stability_threshold,
            "min_long_return": min_long_return,
            "min_short_return": min_short_return,
            "enable_segment_test": enable_segment_test,
            "test_set_ratio": test_set_ratio,
            "max_complexity": max_complexity,
            "transaction_fee": transaction_fee,
            "min_trade_return": min_trade_return
        }
        
        # 显示确认消息
        QMessageBox.information(
            self,
            "设置已应用",
            "因子条件设置已应用。这些设置将在下次启动网格搜索时生效。"
        )
        
        # 记录日志
        self.log_message(f"已应用因子条件设置: IC阈值={ic_threshold}, 稳定性阈值={stability_threshold}, " +
                         f"做多收益要求={min_long_return}, 做空收益要求={min_short_return}, " +
                         f"最大复杂度={max_complexity}, 交易手续费={transaction_fee}%, " +
                         f"单次交易最小收益={min_trade_return}%")
    
    def apply_grid_params(self):
        """应用网格参数设置"""
        # 获取固定参数值
        p_crossover = self.p_crossover_spinbox.value()
        p_subtree_mutation = self.p_subtree_mutation_spinbox.value()
        p_hoist_mutation = self.p_hoist_mutation_spinbox.value()
        p_point_mutation = self.p_point_mutation_spinbox.value()
        
        # 检查概率总和是否不超过1.0
        total_prob = p_crossover + p_subtree_mutation + p_hoist_mutation + p_point_mutation
        if total_prob > 1.0:
            QMessageBox.warning(
                self,
                "概率总和错误",
                f"所有变异概率与交叉概率之和为 {total_prob:.2f}，超过了1.0的限制。\n"
                f"已自动将交叉概率调整为 {1.0 - p_subtree_mutation - p_hoist_mutation - p_point_mutation:.2f}。"
            )
            # 自动调整交叉概率，保持其他概率不变
            p_crossover = 1.0 - p_subtree_mutation - p_hoist_mutation - p_point_mutation
            self.p_crossover_spinbox.setValue(p_crossover)
        
        parsimony_coefficient = self.parsimony_coefficient_spinbox.value()
        
        # 将设置保存到配置中
        # 实际应用中应该更新到PARAM_GRID和FIXED_PARAMS中
        self.grid_params = {
            "forward_period": self.forward_periods,
            "population_size": self.population_sizes,
            "generations": self.generations_values,
            "tournament_size": self.tournament_sizes
        }
        
        self.fixed_params = {
            "p_crossover": p_crossover,
            "p_subtree_mutation": p_subtree_mutation,
            "p_hoist_mutation": p_hoist_mutation,
            "p_point_mutation": p_point_mutation,
            "parsimony_coefficient": parsimony_coefficient
        }
        
        # 计算更新后的组合数
        self.update_combinations_count()
        
        # 显示确认消息
        count = (
            len(self.forward_periods) *
            len(self.population_sizes) *
            len(self.generations_values) *
            len(self.tournament_sizes)
        )
        
        QMessageBox.information(
            self,
            "设置已应用",
            f"网格参数设置已应用。将生成{count}种参数组合。这些设置将在下次启动网格搜索时生效。"
        )
        
        # 记录日志
        self.log_message(f"已应用网格参数设置: 预测周期={self.forward_periods}, " +
                         f"种群大小={self.population_sizes}, 进化代数={self.generations_values}, " +
                         f"锦标赛大小={self.tournament_sizes}, 总组合数={count}")

    def show_param_help(self, title, help_text):
        """显示参数帮助对话框"""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle(f"参数说明 - {title}")
        help_dialog.setMinimumSize(550, 400)
        
        layout = QVBoxLayout(help_dialog)
        
        # 创建文本浏览器显示格式化文本
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setStyleSheet("font-size: 12pt;")
        text_browser.setText(help_text)
        
        layout.addWidget(text_browser)
        
        # 添加确定按钮
        ok_button = QPushButton("确定")
        ok_button.setMinimumHeight(35)
        ok_button.clicked.connect(help_dialog.accept)
        
        layout.addWidget(ok_button)
        
        help_dialog.exec()

    def remove_param_value(self, param_type):
        """删除参数值"""
        if param_type == "forward_period":
            values_list = self.forward_periods
            label = self.forward_period_label
            title = "删除预测周期值"
        elif param_type == "population_size":
            values_list = self.population_sizes
            label = self.population_size_label
            title = "删除种群大小值"
        elif param_type == "generations":
            values_list = self.generations_values
            label = self.generations_label
            title = "删除进化代数值"
        elif param_type == "tournament_size":
            values_list = self.tournament_sizes
            label = self.tournament_size_label
            title = "删除锦标赛大小值"
        else:
            return
            
        # 如果列表为空或只有一个元素，不允许删除
        if len(values_list) <= 1:
            QMessageBox.warning(
                self,
                "无法删除",
                "至少需要保留一个参数值。"
            )
            return
            
        # 创建选择对话框
        values_str = [str(x) for x in sorted(values_list)]
        value, ok = QInputDialog.getItem(
            self,
            title,
            "选择要删除的值:",
            values_str,
            0,
            False
        )
        
        if ok and value:
            # 转换回原始类型
            if param_type == "forward_period":
                val = int(value)
            elif param_type == "population_size":
                val = int(value)
            elif param_type == "generations":
                val = int(value)
            elif param_type == "tournament_size":
                val = int(value)
                
            # 从列表中删除
            if val in values_list:
                values_list.remove(val)
                
                # 更新标签显示
                values_str = ", ".join(map(str, sorted(values_list)))
                label.setText(f"已选: {values_str}")
                
                # 更新组合数
                self.update_combinations_count()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格，更接近Windows 11
    ui = GridSearchUI()
    sys.exit(app.exec()) 