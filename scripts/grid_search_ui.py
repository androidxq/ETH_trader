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
                             QDialog, QVBoxLayout, QTextBrowser, QHeaderView)
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
    finished_signal = pyqtSignal()  # 发送完成信号

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
        self.factor_type = "量价获利因子"  # 默认为量价获利因子

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
            factor_settings = self.custom_factor_settings if self.custom_factor_settings else {}

            # 获取因子类型
            self.factor_type = factor_settings.get("factor_type", "量价获利因子")

            # 为不同因子类型设置推荐函数集
            function_sets = {
                "量价获利因子": ["add", "sub", "mul", "div", "log", "sqrt", "square"],
                "支撑阻力因子": ["add", "sub", "mul", "div", "max", "min", "abs", "if_then_else"],
                "趋势动能因子": ["add", "sub", "mul", "div", "pow", "exp", "sqrt"],
                "波动率因子": ["add", "sub", "mul", "div", "sqrt", "abs", "square"],
                "流动性因子": ["add", "sub", "mul", "div", "log", "sqrt"]
            }

            # 根据因子类型更新函数集
            if "function_set" not in fixed_params or not fixed_params["function_set"]:
                fixed_params["function_set"] = function_sets.get(self.factor_type, ["add", "sub", "mul", "div"])

            # 根据因子类型调整窗口大小
            window_sizes = {
                "量价获利因子": [5, 10, 20, 50, 100],
                "支撑阻力因子": [5, 10, 20, 50, 100, 200],
                "趋势动能因子": [10, 20, 50, 100, 200],
                "波动率因子": [5, 10, 20, 50],
                "流动性因子": [3, 5, 10, 20, 50]
            }

            # 根据因子类型更新窗口大小
            if "windows" not in fixed_params or not fixed_params["windows"]:
                fixed_params["windows"] = window_sizes.get(self.factor_type, [5, 10, 20, 50])

            # 为支撑阻力因子添加特殊的反弹检测功能
            if self.factor_type == "支撑阻力因子":
                # 添加支撑阻力特殊参数
                special_sr_params = {
                    "detect_bounce": True,
                    "min_bounce_percentage": 0.2,
                    "price_level_importance": 0.8,
                    "volume_confirmation": True,
                    "pattern_recognition": True
                }

                # 合并到固定参数中
                fixed_params.update(special_sr_params)

                # 记录特殊参数
                self.update_signal.emit({
                    'type': 'info',
                    'message': f"为支撑阻力因子启用特殊功能:\n"
                               f"- 反弹检测\n"
                               f"- 价格水平重要性评估\n"
                               f"- 成交量确认\n"
                               f"- 形态识别"
                })

            # 输出调整后的参数
            self.update_signal.emit({
                'type': 'info',
                'message': f"为 {self.factor_type} 设置特定参数:\n"
                           f"函数集: {fixed_params['function_set']}\n"
                           f"窗口大小: {fixed_params['windows']}"
            })

            # 从配置中生成网格参数组合
            forward_periods = param_grid.get("forward_period", [12])
            populations = param_grid.get("population_size", [1000])
            generations = param_grid.get("generations", [100])
            tournament_sizes = param_grid.get("tournament_size", [20])

            # 获取因子条件设置中的交易手续费和最小交易收益参数
            transaction_fee = factor_settings.get("transaction_fee", 0.1)
            min_trade_return = factor_settings.get("min_trade_return", 0.3)

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
                                    "factor_type": self.factor_type,  # 添加因子类型参数
                                    "ic_threshold": self.custom_factor_settings.get("ic_threshold", 0.05),
                                    "stability_threshold": self.custom_factor_settings.get("stability_threshold", 0.3),
                                    "min_long_return": self.custom_factor_settings.get("min_long_return", 0.5),
                                    "min_short_return": self.custom_factor_settings.get("min_short_return", -0.5),
                                    "enable_segment_test": self.custom_factor_settings.get("enable_segment_test", True),
                                    "test_set_ratio": self.custom_factor_settings.get("test_set_ratio", 0.3),
                                    "max_complexity": self.custom_factor_settings.get("max_complexity", 20),
                                    "transaction_fee": self.custom_factor_settings.get("transaction_fee", 0.1),
                                    "min_trade_return": self.custom_factor_settings.get("min_trade_return", 0.3)
                                })
                            # 如果没有自定义因子条件，仍然添加默认的交易手续费和最小交易收益以及因子类型
                            else:
                                params.update({
                                    "factor_type": self.factor_type,
                                    "transaction_fee": transaction_fee,
                                    "min_trade_return": min_trade_return
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
                    'message': f"正在处理第 {batch_idx + 1}/{num_batches} 批 (组合 {start_idx + 1} 到 {end_idx})"
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
                                pool.join(0.5)  # 短暂等待进程终止
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
                                    try:
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
                            except StopIteration:
                                # 所有结果都已处理完
                                break
                            except Exception as e:
                                self.log_message(f"处理结果时出错: {str(e)}")
                                continue

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
                intermediate_file = f"{self.searcher.results_dir}/grid_search_intermediate_batch_{batch_idx + 1}_{num_batches}_{self.searcher.timestamp}.pkl"
                with open(intermediate_file, 'wb') as f:
                    pickle.dump(results, f)

                self.update_signal.emit({
                    'type': 'batch_end',
                    'batch_idx': batch_idx,
                    'total_batches': num_batches,
                    'message': f"已完成批次 {batch_idx + 1}/{num_batches} 的处理"
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
        """初始化UI"""
        super().__init__(parent)

        # 初始化属性
        self.worker = None
        self.paused = False
        self.running = False
        self.process_info = {}
        self.active_processes = 0

        # 内存和进程监控计时器
        self.memory_timer = QTimer(self)
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.setInterval(3000)  # 3秒刷新一次

        self.process_timer = QTimer(self)
        self.process_timer.timeout.connect(self.update_process_resources)
        self.process_timer.setInterval(3000)  # 3秒刷新一次

        # 初始化网格参数
        self.forward_periods = [12]  # 预测周期
        self.population_sizes = [1000]  # 种群大小
        self.generations_values = [100]  # 进化代数
        self.tournament_sizes = [20]  # 锦标赛大小

        # 初始化固定参数
        self.fixed_params = FIXED_PARAMS.copy()
        self.grid_params = PARAM_GRID.copy()

        # 保存找到的因子
        self.factors_found = []

        # 初始化默认的因子设置
        self.factor_settings = {
            "factor_type": "量价获利因子",  # 默认选择量价获利因子
            "ic_threshold": 0.05,
            "stability_threshold": 0.3,
            "min_long_return": 0.5,
            "min_short_return": -0.5,
            "enable_segment_test": True,
            "test_set_ratio": 0.3,
            "max_complexity": 25,
            "transaction_fee": 0.1,
            "min_trade_return": 0.3
        }

        # 初始化UI
        self.init_ui()

        # 更新组合数量显示
        self.update_combinations_count()

        # 初始化显示第一个因子类型的描述和参数
        self.update_factor_type_description()

        # 记录日志 - 在初始化UI和更新因子描述后调用，确保所有UI组件都已创建
        self.log_message("系统已启动，等待开始网格搜索...")

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle('ETH因子网格搜索')
        self.resize(1600, 900)  # 使用更宽的窗口，适合三列布局

        # 创建主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)  # 增加主布局的边距

        # 创建左侧布局 - 参数设置区域
        left_widget = QWidget()

        # 创建三列布局
        self.left_layout = QHBoxLayout(left_widget)
        self.left_layout.setSpacing(12)  # 增加列之间的间距
        self.left_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距使布局更紧凑

        # 创建三列的垂直布局
        column1_layout = QVBoxLayout()
        column2_layout = QVBoxLayout()
        column3_layout = QVBoxLayout()

        column1_layout.setSpacing(10)  # 紧凑的垂直间距
        column2_layout.setSpacing(10)
        column3_layout.setSpacing(10)

        # 将三列添加到左侧布局中
        self.left_layout.addLayout(column1_layout, 1)  # 第一列
        self.left_layout.addLayout(column2_layout, 1)  # 第二列
        self.left_layout.addLayout(column3_layout, 1)  # 第三列

        # 设置全局样式
        left_widget.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 12px;
                margin-top: 10px;
            }
            QLabel {
                min-height: 22px;
                font-size: 11px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                min-height: 24px;
                padding: 1px;
                font-size: 11px;
            }
            QPushButton {
                min-height: 24px;
                padding: 3px;
                font-size: 11px;
            }
            QCheckBox {
                min-height: 22px;
                font-size: 11px;
            }
            QTextEdit {
                font-size: 11px;
                padding: 3px;
            }
        """)

        # 接下来我们将创建各个组件并分配到三列中

        # ==== 第一列组件 ====

        # 因子类型选择区域
        self.factor_type_group = QGroupBox("因子类型")
        factor_type_layout = QVBoxLayout()
        factor_type_layout.setSpacing(6)

        self.factor_type_combo = QComboBox()
        self.factor_type_combo.addItems([
            "量价获利因子", "支撑阻力因子", "趋势动能因子",
            "波动率因子", "流动性因子"
        ])
        self.factor_type_combo.currentIndexChanged.connect(self.update_factor_type_description)

        self.factor_description = QTextEdit()
        self.factor_description.setReadOnly(True)
        self.factor_description.setMaximumHeight(80)  # 降低高度使布局更紧凑

        factor_type_layout.addWidget(QLabel("选择因子类型:"))
        factor_type_layout.addWidget(self.factor_type_combo)
        factor_type_layout.addWidget(QLabel("因子类型说明:"))
        factor_type_layout.addWidget(self.factor_description)

        # 添加函数集选择器
        self.function_set_label = QLabel("推荐函数集:")
        self.function_set_layout = QHBoxLayout()
        self.function_set_layout.addWidget(self.function_set_label)

        factor_type_layout.addLayout(self.function_set_layout)

        self.factor_type_group.setLayout(factor_type_layout)
        column1_layout.addWidget(self.factor_type_group)

        # 因子条件设置区域 - 第一部分 (IC阈值、稳定性阈值、多头最小收益)
        self.factor_settings_group1 = QGroupBox("因子条件设置")
        factor_settings_layout1 = QGridLayout()
        factor_settings_layout1.setHorizontalSpacing(8)
        factor_settings_layout1.setVerticalSpacing(6)
        factor_settings_layout1.setContentsMargins(8, 12, 8, 8)

        # IC阈值设置
        row = 0
        self.ic_threshold_label = QLabel("IC阈值:")
        self.ic_threshold_spinbox = QDoubleSpinBox()
        self.ic_threshold_spinbox.setRange(0.01, 1.0)
        self.ic_threshold_spinbox.setSingleStep(0.01)
        self.ic_threshold_spinbox.setValue(0.05)
        self.ic_threshold_help = QPushButton("?")
        self.ic_threshold_help.setMaximumWidth(20)
        self.ic_threshold_help.clicked.connect(lambda: self.show_param_help("IC阈值",
                                                                            "IC阈值(Information Coefficient)是衡量因子预测能力的指标。\n\n"
                                                                            "数值范围: 0.01-1.0\n"
                                                                            "推荐设置: \n"
                                                                            "- 量价获利因子: 0.05\n"
                                                                            "- 支撑阻力因子: 0.03\n"
                                                                            "- 趋势动能因子: 0.04\n"
                                                                            "- 波动率因子: 0.03\n"
                                                                            "- 流动性因子: 0.04\n\n"
                                                                            "该阈值越高，筛选的因子预测能力越强，但符合条件的因子数量可能减少。"))

        factor_settings_layout1.addWidget(self.ic_threshold_label, row, 0)
        factor_settings_layout1.addWidget(self.ic_threshold_spinbox, row, 1)
        factor_settings_layout1.addWidget(self.ic_threshold_help, row, 2)

        # 稳定性阈值设置
        row += 1
        self.stability_threshold_label = QLabel("稳定性阈值:")
        self.stability_threshold_spinbox = QDoubleSpinBox()
        self.stability_threshold_spinbox.setRange(0.1, 1.0)
        self.stability_threshold_spinbox.setSingleStep(0.05)
        self.stability_threshold_spinbox.setValue(0.3)
        self.stability_threshold_help = QPushButton("?")
        self.stability_threshold_help.setMaximumWidth(20)
        self.stability_threshold_help.clicked.connect(lambda: self.show_param_help("稳定性阈值",
                                                                                   "稳定性阈值衡量因子在不同时间段的表现一致性。\n\n"
                                                                                   "数值范围: 0.1-1.0\n"
                                                                                   "推荐设置: \n"
                                                                                   "- 量价获利因子: 0.3\n"
                                                                                   "- 支撑阻力因子: 0.4\n"
                                                                                   "- 趋势动能因子: 0.35\n"
                                                                                   "- 波动率因子: 0.25\n"
                                                                                   "- 流动性因子: 0.3\n\n"
                                                                                   "该值越高，筛选的因子表现越稳定，但可能会错过一些在特定市场环境表现良好的因子。"))

        factor_settings_layout1.addWidget(self.stability_threshold_label, row, 0)
        factor_settings_layout1.addWidget(self.stability_threshold_spinbox, row, 1)
        factor_settings_layout1.addWidget(self.stability_threshold_help, row, 2)

        # 多头最小收益设置
        row += 1
        self.min_long_return_label = QLabel("多头最小收益(%):")
        self.min_long_return_spinbox = QDoubleSpinBox()
        self.min_long_return_spinbox.setRange(0.1, 10.0)
        self.min_long_return_spinbox.setSingleStep(0.1)
        self.min_long_return_spinbox.setValue(0.5)
        self.min_long_return_help = QPushButton("?")
        self.min_long_return_help.setMaximumWidth(20)
        self.min_long_return_help.clicked.connect(lambda: self.show_param_help("多头最小收益",
                                                                               "多头最小收益要求因子在多头方向上的最小收益率。\n\n"
                                                                               "数值范围: 0.1%-10.0%\n"
                                                                               "推荐设置: \n"
                                                                               "- 量价获利因子: 0.5%\n"
                                                                               "- 支撑阻力因子: 0.3%\n"
                                                                               "- 趋势动能因子: 0.6%\n"
                                                                               "- 波动率因子: 0.4%\n"
                                                                               "- 流动性因子: 0.45%\n\n"
                                                                               "该值越高，筛选的因子盈利能力越强，但可能会错过一些小幅盈利但更稳定的因子。"))

        factor_settings_layout1.addWidget(self.min_long_return_label, row, 0)
        factor_settings_layout1.addWidget(self.min_long_return_spinbox, row, 1)
        factor_settings_layout1.addWidget(self.min_long_return_help, row, 2)

        # 空头最小收益设置
        row += 1
        self.min_short_return_label = QLabel("空头最小收益(%):")
        self.min_short_return_spinbox = QDoubleSpinBox()
        self.min_short_return_spinbox.setRange(-10.0, 0.0)
        self.min_short_return_spinbox.setSingleStep(0.1)
        self.min_short_return_spinbox.setValue(-0.5)
        self.min_short_return_help = QPushButton("?")
        self.min_short_return_help.setMaximumWidth(20)
        self.min_short_return_help.clicked.connect(lambda: self.show_param_help("空头最小收益",
                                                                                "空头最小收益要求因子在空头方向上的最小收益率。\n\n"
                                                                                "数值范围: -10.0%-0.0%\n"
                                                                                "推荐设置: \n"
                                                                                "- 量价获利因子: -0.5%\n"
                                                                                "- 支撑阻力因子: -0.3%\n"
                                                                                "- 趋势动能因子: -0.6%\n"
                                                                                "- 波动率因子: -0.4%\n"
                                                                                "- 流动性因子: -0.45%\n\n"
                                                                                "该值越低(绝对值越大)，筛选的因子空头表现越好，但可能会错过一些小幅空头盈利但更稳定的因子。"))

        factor_settings_layout1.addWidget(self.min_short_return_label, row, 0)
        factor_settings_layout1.addWidget(self.min_short_return_spinbox, row, 1)
        factor_settings_layout1.addWidget(self.min_short_return_help, row, 2)

        self.factor_settings_group1.setLayout(factor_settings_layout1)
        column1_layout.addWidget(self.factor_settings_group1)

        # 创建交易手续费设置
        self.transaction_fee_label = QLabel("交易手续费(%):")
        self.transaction_fee_spinbox = QDoubleSpinBox()
        self.transaction_fee_spinbox.setRange(0.01, 1.0)
        self.transaction_fee_spinbox.setSingleStep(0.01)
        self.transaction_fee_spinbox.setValue(0.1)
        self.transaction_fee_help = QPushButton("?")
        self.transaction_fee_help.setMaximumWidth(25)
        self.transaction_fee_help.clicked.connect(lambda: self.show_param_help("交易手续费",
                                                                               "交易手续费率，影响净收益计算。\n\n"
                                                                               "数值范围: 0.01%-1.0%\n"
                                                                               "推荐设置: 0.1% (通常保持一致)\n\n"
                                                                               "该值应根据实际交易环境设置，过低会导致过度乐观的回测结果，过高会过滤掉一些有潜力的因子。"))

        # 创建最小交易收益设置
        self.min_trade_return_label = QLabel("最小交易收益(%):")
        self.min_trade_return_spinbox = QDoubleSpinBox()
        self.min_trade_return_spinbox.setRange(0.1, 5.0)
        self.min_trade_return_spinbox.setSingleStep(0.1)
        self.min_trade_return_spinbox.setValue(0.3)
        self.min_trade_return_help = QPushButton("?")
        self.min_trade_return_help.setMaximumWidth(25)
        self.min_trade_return_help.clicked.connect(lambda: self.show_param_help("最小交易收益",
                                                                                "单次交易的最小期望收益，低于此值的交易信号将被过滤。\n\n"
                                                                                "数值范围: 0.1%-5.0%\n"
                                                                                "推荐设置: \n"
                                                                                "- 量价获利因子: 0.3%\n"
                                                                                "- 支撑阻力因子: 0.2%\n"
                                                                                "- 趋势动能因子: 0.4%\n"
                                                                                "- 波动率因子: 0.35%\n"
                                                                                "- 流动性因子: 0.3%\n\n"
                                                                                "该值越高，筛选的交易信号质量越高，但可能导致交易频率降低。"))

        # 创建分段测试设置
        self.enable_segment_test_label = QLabel("启用分段测试:")
        self.enable_segment_test_checkbox = QCheckBox()
        self.enable_segment_test_checkbox.setChecked(True)
        self.enable_segment_test_help = QPushButton("?")
        self.enable_segment_test_help.setMaximumWidth(25)
        self.enable_segment_test_help.clicked.connect(lambda: self.show_param_help("启用分段测试",
                                                                                   "是否在不同的市场环境下评估因子表现。\n\n"
                                                                                   "推荐设置: 启用\n\n"
                                                                                   "启用后将在不同的时间段和市场环境中测试因子表现，有助于筛选出更稳健的因子，但会增加计算量。"))

        # 创建最大复杂度设置
        self.max_complexity_label = QLabel("最大复杂度:")
        self.max_complexity_spinbox = QSpinBox()
        self.max_complexity_spinbox.setRange(5, 50)
        self.max_complexity_spinbox.setSingleStep(1)
        self.max_complexity_spinbox.setValue(20)
        self.max_complexity_help = QPushButton("?")
        self.max_complexity_help.setMaximumWidth(25)
        self.max_complexity_help.clicked.connect(lambda: self.show_param_help("最大复杂度",
                                                                              "因子表达式的最大复杂度限制。\n\n"
                                                                              "数值范围: 5-50\n"
                                                                              "推荐设置: 20 (可根据因子类型调整)\n\n"
                                                                              "该值越高，允许的因子表达式越复杂，可能发现更精细的模式，但也可能导致过拟合风险增加。"))

        # 添加因子条件设置到布局
        row = 0
        self.factor_settings_layout = factor_settings_layout1  # 添加这行来保持兼容性
        self.factor_settings_layout.addWidget(self.ic_threshold_label, row, 0)
        self.factor_settings_layout.addWidget(self.ic_threshold_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.ic_threshold_help, row, 2)

        row += 1
        self.factor_settings_layout.addWidget(self.stability_threshold_label, row, 0)
        self.factor_settings_layout.addWidget(self.stability_threshold_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.stability_threshold_help, row, 2)

        row += 1
        self.factor_settings_layout.addWidget(self.min_long_return_label, row, 0)
        self.factor_settings_layout.addWidget(self.min_long_return_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.min_long_return_help, row, 2)

        row += 1
        self.factor_settings_layout.addWidget(self.min_short_return_label, row, 0)
        self.factor_settings_layout.addWidget(self.min_short_return_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.min_short_return_help, row, 2)

        row += 1
        self.factor_settings_layout.addWidget(self.transaction_fee_label, row, 0)
        self.factor_settings_layout.addWidget(self.transaction_fee_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.transaction_fee_help, row, 2)

        row += 1
        self.factor_settings_layout.addWidget(self.min_trade_return_label, row, 0)
        self.factor_settings_layout.addWidget(self.min_trade_return_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.min_trade_return_help, row, 2)

        row += 1
        self.factor_settings_layout.addWidget(self.enable_segment_test_label, row, 0)
        self.factor_settings_layout.addWidget(self.enable_segment_test_checkbox, row, 1)
        self.factor_settings_layout.addWidget(self.enable_segment_test_help, row, 2)

        # 添加测试集比例控件到布局
        row += 1
        self.test_set_ratio_label = QLabel("测试集比例:")
        self.test_set_ratio_spinbox = QDoubleSpinBox()
        self.test_set_ratio_spinbox.setRange(0.1, 0.5)
        self.test_set_ratio_spinbox.setSingleStep(0.05)
        self.test_set_ratio_spinbox.setValue(0.3)
        self.test_set_ratio_help = QPushButton("?")
        self.test_set_ratio_help.setMaximumWidth(25)
        self.test_set_ratio_help.clicked.connect(lambda: self.show_param_help("测试集比例",
                                                                              "用于评估因子表现的数据比例。\n\n"
                                                                              "数值范围: 0.1-0.5\n"
                                                                              "推荐设置: 0.3 (30%的数据用于测试)\n\n"
                                                                              "较高的比例有助于更好地评估因子在不同市场环境中的表现，但会减少训练数据量。"))
        self.factor_settings_layout.addWidget(self.test_set_ratio_label, row, 0)
        self.factor_settings_layout.addWidget(self.test_set_ratio_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.test_set_ratio_help, row, 2)

        row += 1
        self.factor_settings_layout.addWidget(self.max_complexity_label, row, 0)
        self.factor_settings_layout.addWidget(self.max_complexity_spinbox, row, 1)
        self.factor_settings_layout.addWidget(self.max_complexity_help, row, 2)

        # 添加应用按钮
        row += 1
        self.apply_factor_settings_btn = QPushButton("应用因子条件")
        self.apply_factor_settings_btn.clicked.connect(self.apply_factor_settings)
        self.factor_settings_layout.addWidget(self.apply_factor_settings_btn, row, 0, 1, 3)

        # 注释掉这两行代码，因为self.factor_settings_group未定义
        # self.factor_settings_group.setLayout(self.factor_settings_layout)
        # column2_layout.addWidget(self.factor_settings_group)

        # ==== 第二列组件 ====

        # 添加支撑阻力因子特殊配置区域
        self.support_resistance_config_group = QGroupBox("支撑阻力因子特殊配置")
        self.support_resistance_config_layout = QGridLayout()
        self.support_resistance_config_layout.setHorizontalSpacing(8)
        self.support_resistance_config_layout.setVerticalSpacing(6)
        self.support_resistance_config_layout.setContentsMargins(8, 12, 8, 8)

        # 反弹检测设置
        row = 0
        self.detect_bounce_label = QLabel("启用反弹检测:")
        self.detect_bounce_checkbox = QCheckBox()
        self.detect_bounce_checkbox.setChecked(True)
        self.detect_bounce_help = QPushButton("?")
        self.detect_bounce_help.setMaximumWidth(20)
        self.detect_bounce_help.clicked.connect(lambda: self.show_param_help("启用反弹检测",
                                                                             "检测价格在支撑阻力位的反弹行为。\n\n"
                                                                             "推荐设置: 启用\n\n"
                                                                             "此功能可识别价格在关键水平的反转行为，是支撑阻力因子的核心功能。"))

        self.support_resistance_config_layout.addWidget(self.detect_bounce_label, row, 0)
        self.support_resistance_config_layout.addWidget(self.detect_bounce_checkbox, row, 1)
        self.support_resistance_config_layout.addWidget(self.detect_bounce_help, row, 2)

        # 最小反弹幅度设置
        row += 1
        self.min_bounce_percentage_label = QLabel("最小反弹幅度(%):")
        self.min_bounce_percentage_spinbox = QDoubleSpinBox()
        self.min_bounce_percentage_spinbox.setRange(0.1, 5.0)
        self.min_bounce_percentage_spinbox.setSingleStep(0.1)
        self.min_bounce_percentage_spinbox.setValue(0.2)
        self.min_bounce_percentage_help = QPushButton("?")
        self.min_bounce_percentage_help.setMaximumWidth(20)
        self.min_bounce_percentage_help.clicked.connect(lambda: self.show_param_help("最小反弹幅度",
                                                                                     "价格在支撑阻力位反弹的最小幅度要求。\n\n"
                                                                                     "数值范围: 0.1%-5.0%\n"
                                                                                     "推荐设置: 0.2-0.3%\n\n"
                                                                                     "此参数定义了有效反弹的最小标准，过小的值可能导致误报，过大的值可能错过一些轻微但有效的反弹机会。"))

        self.support_resistance_config_layout.addWidget(self.min_bounce_percentage_label, row, 0)
        self.support_resistance_config_layout.addWidget(self.min_bounce_percentage_spinbox, row, 1)
        self.support_resistance_config_layout.addWidget(self.min_bounce_percentage_help, row, 2)

        # 价格水平重要性设置
        row += 1
        self.price_level_importance_label = QLabel("价格水平重要性:")
        self.price_level_importance_spinbox = QDoubleSpinBox()
        self.price_level_importance_spinbox.setRange(0.1, 1.0)
        self.price_level_importance_spinbox.setSingleStep(0.1)
        self.price_level_importance_spinbox.setValue(0.8)
        self.price_level_importance_help = QPushButton("?")
        self.price_level_importance_help.setMaximumWidth(20)
        self.price_level_importance_help.clicked.connect(lambda: self.show_param_help("价格水平重要性",
                                                                                      "对历史价格水平重要性的权重系数。\n\n"
                                                                                      "数值范围: 0.1-1.0\n"
                                                                                      "推荐设置: 0.8\n\n"
                                                                                      "此参数决定如何评估价格水平的重要性，值越高，历史上交易更活跃的价格水平权重越大。"))

        self.support_resistance_config_layout.addWidget(self.price_level_importance_label, row, 0)
        self.support_resistance_config_layout.addWidget(self.price_level_importance_spinbox, row, 1)
        self.support_resistance_config_layout.addWidget(self.price_level_importance_help, row, 2)

        # 成交量确认设置
        row += 1
        self.volume_confirmation_label = QLabel("成交量确认:")
        self.volume_confirmation_checkbox = QCheckBox()
        self.volume_confirmation_checkbox.setChecked(True)
        self.volume_confirmation_help = QPushButton("?")
        self.volume_confirmation_help.setMaximumWidth(20)
        self.volume_confirmation_help.clicked.connect(lambda: self.show_param_help("成交量确认",
                                                                                   "使用成交量变化来确认价格反转。\n\n"
                                                                                   "推荐设置: 启用\n\n"
                                                                                   "此功能检查支撑阻力位处的成交量模式，有效的反转通常伴随着特定的成交量特征，比如支撑位放量反弹、阻力位缩量回落。"))

        self.support_resistance_config_layout.addWidget(self.volume_confirmation_label, row, 0)
        self.support_resistance_config_layout.addWidget(self.volume_confirmation_checkbox, row, 1)
        self.support_resistance_config_layout.addWidget(self.volume_confirmation_help, row, 2)

        # 价格形态识别设置
        row += 1
        self.pattern_recognition_label = QLabel("价格形态识别:")
        self.pattern_recognition_checkbox = QCheckBox()
        self.pattern_recognition_checkbox.setChecked(True)
        self.pattern_recognition_help = QPushButton("?")
        self.pattern_recognition_help.setMaximumWidth(20)
        self.pattern_recognition_help.clicked.connect(lambda: self.show_param_help("价格形态识别",
                                                                                   "检测典型的反转K线形态，如锤子线、吞没形态等。\n\n"
                                                                                   "推荐设置: 启用\n\n"
                                                                                   "此功能能够识别在支撑阻力位出现的特殊K线形态，这些形态往往是价格反转的重要信号。"))

        self.support_resistance_config_layout.addWidget(self.pattern_recognition_label, row, 0)
        self.support_resistance_config_layout.addWidget(self.pattern_recognition_checkbox, row, 1)
        self.support_resistance_config_layout.addWidget(self.pattern_recognition_help, row, 2)

        # 添加应用按钮
        row += 1
        self.apply_sr_config_btn = QPushButton("应用支撑阻力特殊配置")
        self.apply_sr_config_btn.clicked.connect(self.apply_sr_config)
        self.support_resistance_config_layout.addWidget(self.apply_sr_config_btn, row, 0, 1, 3)

        self.support_resistance_config_group.setLayout(self.support_resistance_config_layout)
        column2_layout.addWidget(self.support_resistance_config_group)
        self.support_resistance_config_group.setVisible(False)  # 默认隐藏

        # ==== 第三列组件 ====

        # 网格参数设置区域
        self.grid_params_group = QGroupBox("网格参数设置")
        self.grid_params_layout = QGridLayout()
        self.grid_params_layout.setHorizontalSpacing(8)
        self.grid_params_layout.setVerticalSpacing(6)
        self.grid_params_layout.setContentsMargins(8, 12, 8, 8)

        # 预测周期设置
        row = 0
        self.grid_params_layout.addWidget(QLabel("预测周期:"), row, 0)
        self.forward_period_label = QLabel(f"已选: {', '.join(map(str, self.forward_periods))}")

        control_layout = QHBoxLayout()
        forward_period_add_btn = QPushButton("+")
        forward_period_add_btn.setMaximumWidth(25)
        forward_period_add_btn.clicked.connect(lambda: self.add_period_value("forward_period"))
        forward_period_remove_btn = QPushButton("-")
        forward_period_remove_btn.setMaximumWidth(25)
        forward_period_remove_btn.clicked.connect(lambda: self.remove_param_value("forward_period"))
        control_layout.addWidget(forward_period_add_btn)
        control_layout.addWidget(forward_period_remove_btn)

        self.grid_params_layout.addWidget(self.forward_period_label, row, 1)
        self.grid_params_layout.addLayout(control_layout, row, 2)

        # 种群大小设置
        row += 1
        self.grid_params_layout.addWidget(QLabel("种群大小:"), row, 0)
        self.population_size_label = QLabel(f"已选: {', '.join(map(str, self.population_sizes))}")

        control_layout = QHBoxLayout()
        population_size_add_btn = QPushButton("+")
        population_size_add_btn.setMaximumWidth(25)
        population_size_add_btn.clicked.connect(lambda: self.add_int_value("population_size"))
        population_size_remove_btn = QPushButton("-")
        population_size_remove_btn.setMaximumWidth(25)
        population_size_remove_btn.clicked.connect(lambda: self.remove_param_value("population_size"))
        control_layout.addWidget(population_size_add_btn)
        control_layout.addWidget(population_size_remove_btn)

        self.grid_params_layout.addWidget(self.population_size_label, row, 1)
        self.grid_params_layout.addLayout(control_layout, row, 2)

        # 进化代数设置
        row += 1
        self.grid_params_layout.addWidget(QLabel("进化代数:"), row, 0)
        self.generations_label = QLabel(f"已选: {', '.join(map(str, self.generations_values))}")

        control_layout = QHBoxLayout()
        generations_add_btn = QPushButton("+")
        generations_add_btn.setMaximumWidth(25)
        generations_add_btn.clicked.connect(lambda: self.add_int_value("generations"))
        generations_remove_btn = QPushButton("-")
        generations_remove_btn.setMaximumWidth(25)
        generations_remove_btn.clicked.connect(lambda: self.remove_param_value("generations"))
        control_layout.addWidget(generations_add_btn)
        control_layout.addWidget(generations_remove_btn)

        self.grid_params_layout.addWidget(self.generations_label, row, 1)
        self.grid_params_layout.addLayout(control_layout, row, 2)

        # 锦标赛大小设置
        row += 1
        self.grid_params_layout.addWidget(QLabel("锦标赛大小:"), row, 0)
        self.tournament_size_label = QLabel(f"已选: {', '.join(map(str, self.tournament_sizes))}")

        control_layout = QHBoxLayout()
        tournament_size_add_btn = QPushButton("+")
        tournament_size_add_btn.setMaximumWidth(25)
        tournament_size_add_btn.clicked.connect(lambda: self.add_int_value("tournament_size"))
        tournament_size_remove_btn = QPushButton("-")
        tournament_size_remove_btn.setMaximumWidth(25)
        tournament_size_remove_btn.clicked.connect(lambda: self.remove_param_value("tournament_size"))
        control_layout.addWidget(tournament_size_add_btn)
        control_layout.addWidget(tournament_size_remove_btn)

        self.grid_params_layout.addWidget(self.tournament_size_label, row, 1)
        self.grid_params_layout.addLayout(control_layout, row, 2)

        # 显示可能的组合数
        row += 1
        self.grid_params_layout.addWidget(QLabel("可能的组合数:"), row, 0)
        self.combinations_label = QLabel("")
        self.grid_params_layout.addWidget(self.combinations_label, row, 1, 1, 2)

        # 添加遗传算法参数设置
        # 交叉概率
        row += 1
        self.grid_params_layout.addWidget(QLabel("交叉概率:"), row, 0)
        self.p_crossover_spinbox = QDoubleSpinBox()
        self.p_crossover_spinbox.setRange(0.1, 1.0)
        self.p_crossover_spinbox.setSingleStep(0.05)
        self.p_crossover_spinbox.setValue(0.7)
        self.grid_params_layout.addWidget(self.p_crossover_spinbox, row, 1, 1, 2)

        # 子树变异概率
        row += 1
        self.grid_params_layout.addWidget(QLabel("子树变异概率:"), row, 0)
        self.p_subtree_mutation_spinbox = QDoubleSpinBox()
        self.p_subtree_mutation_spinbox.setRange(0.0, 0.5)
        self.p_subtree_mutation_spinbox.setSingleStep(0.05)
        self.p_subtree_mutation_spinbox.setValue(0.15)
        self.grid_params_layout.addWidget(self.p_subtree_mutation_spinbox, row, 1, 1, 2)

        # 提升变异概率
        row += 1
        self.grid_params_layout.addWidget(QLabel("提升变异概率:"), row, 0)
        self.p_hoist_mutation_spinbox = QDoubleSpinBox()
        self.p_hoist_mutation_spinbox.setRange(0.0, 0.5)
        self.p_hoist_mutation_spinbox.setSingleStep(0.05)
        self.p_hoist_mutation_spinbox.setValue(0.1)
        self.grid_params_layout.addWidget(self.p_hoist_mutation_spinbox, row, 1, 1, 2)

        # 点变异概率
        row += 1
        self.grid_params_layout.addWidget(QLabel("点变异概率:"), row, 0)
        self.p_point_mutation_spinbox = QDoubleSpinBox()
        self.p_point_mutation_spinbox.setRange(0.0, 0.5)
        self.p_point_mutation_spinbox.setSingleStep(0.05)
        self.p_point_mutation_spinbox.setValue(0.1)
        self.grid_params_layout.addWidget(self.p_point_mutation_spinbox, row, 1, 1, 2)

        # 简约性系数
        row += 1
        self.grid_params_layout.addWidget(QLabel("简约性系数:"), row, 0)
        self.parsimony_coefficient_spinbox = QDoubleSpinBox()
        self.parsimony_coefficient_spinbox.setRange(0.0001, 0.01)
        self.parsimony_coefficient_spinbox.setSingleStep(0.0001)
        self.parsimony_coefficient_spinbox.setDecimals(5)
        self.parsimony_coefficient_spinbox.setValue(0.001)
        self.grid_params_layout.addWidget(self.parsimony_coefficient_spinbox, row, 1, 1, 2)

        # 添加应用按钮
        row += 1
        self.apply_grid_params_btn = QPushButton("应用网格参数")
        self.apply_grid_params_btn.clicked.connect(self.apply_grid_params)
        self.grid_params_layout.addWidget(self.apply_grid_params_btn, row, 0, 1, 3)

        self.grid_params_group.setLayout(self.grid_params_layout)
        column3_layout.addWidget(self.grid_params_group)

        # 控制按钮区域
        control_group = QGroupBox("操作控制")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)

        # 状态显示
        status_layout = QHBoxLayout()
        self.status_label = QLabel("状态: 就绪")
        self.memory_label = QLabel("内存使用率: 0%")
        self.process_label = QLabel("活跃进程数: 0")
        status_layout.addWidget(self.status_label, 2)
        status_layout.addWidget(self.memory_label, 1)
        status_layout.addWidget(self.process_label, 1)
        control_layout.addLayout(status_layout)

        # 进度条
        progress_layout = QGridLayout()
        progress_layout.addWidget(QLabel("总进度:"), 0, 0)
        self.total_progress_bar = QProgressBar()
        self.total_progress_bar.setRange(0, 100)
        self.total_progress_bar.setValue(0)
        progress_layout.addWidget(self.total_progress_bar, 0, 1)

        progress_layout.addWidget(QLabel("批次进度:"), 1, 0)
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        progress_layout.addWidget(self.batch_progress_bar, 1, 1)

        control_layout.addLayout(progress_layout)

        # 控制按钮
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)
        buttons_layout.setContentsMargins(5, 5, 5, 5)

        self.start_button = QPushButton("开始搜索")
        self.start_button.setMinimumHeight(45)
        self.start_button.setFont(QFont("微软雅黑", 11, QFont.Weight.Bold))
        self.start_button.clicked.connect(self.start_search)

        self.pause_button = QPushButton("暂停")
        self.pause_button.setMinimumHeight(45)
        self.pause_button.setFont(QFont("微软雅黑", 11))
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.stop_button = QPushButton("停止")
        self.stop_button.setMinimumHeight(45)
        self.stop_button.setFont(QFont("微软雅黑", 11))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_search)

        self.report_button = QPushButton("查看报告")
        self.report_button.setMinimumHeight(45)
        self.report_button.setFont(QFont("微软雅黑", 11))
        self.report_button.clicked.connect(self.open_report)

        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.report_button)

        control_layout.addLayout(buttons_layout)

        control_group.setLayout(control_layout)
        column3_layout.addWidget(control_group)

        # ==== 创建右侧布局 - 结果区域 ====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)  # 设置垂直间距
        right_layout.setContentsMargins(10, 10, 10, 10)  # 设置内边距

        # 设置右侧控件样式
        right_widget.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                padding-top: 15px;
                margin-top: 15px;
            }
            QLabel {
                min-height: 25px;
                font-size: 12px;
            }
            QProgressBar {
                min-height: 22px;
                text-align: center;
                font-size: 11px;
            }
            QTableWidget {
                font-size: 12px;
            }
        """)

        # 创建进程表
        process_group = QGroupBox("进程监控")
        process_layout = QVBoxLayout()

        self.process_table = QTableWidget()
        self.process_table.setColumnCount(5)
        self.process_table.setHorizontalHeaderLabels([
            "进程ID", "CPU使用率", "内存使用", "运行时间", "状态"
        ])
        self.process_table.horizontalHeader().setStretchLastSection(True)
        self.process_table.setMinimumHeight(120)
        process_layout.addWidget(self.process_table)

        process_group.setLayout(process_layout)
        right_layout.addWidget(process_group)

        # 创建Tabs - 结果和日志
        results_tabs = QTabWidget()

        # 创建结果Tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # 创建结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(10)
        self.results_table.setHorizontalHeaderLabels([
            "预测周期", "表达式", "IC值", "稳定性",
            "多头收益", "多头净收益", "多头有效率",
            "空头收益", "空头净收益", "空头有效率"
        ])

        # 设置表格样式
        self.results_table.setStyleSheet("""
            QTableWidget {
                font-size: 12px;
                gridline-color: #d0d0d0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                font-weight: bold;
                padding: 6px;
                border: 1px solid #d0d0d0;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }
        """)

        # 设置行高
        self.results_table.verticalHeader().setDefaultSectionSize(32)

        # 设置自动调整列宽
        self.results_table.horizontalHeader().setStretchLastSection(True)

        # 允许表格内容自动调整
        for col in range(2, self.results_table.columnCount()):
            self.results_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        # 表达式列可伸展
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        results_layout.addWidget(self.results_table)
        results_tabs.addTab(results_tab, "因子结果")

        # 创建日志Tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)

        # 创建日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            font-family: "Consolas", "Courier New", monospace;
            font-size: 12px;
            line-height: 1.5;
        """)
        log_layout.addWidget(self.log_text)
        results_tabs.addTab(log_tab, "完整日志")

        # 添加状态日志
        status_log_tab = QWidget()
        status_log_layout = QVBoxLayout(status_log_tab)
        self.status_log_text = QTextEdit()
        self.status_log_text.setReadOnly(True)
        self.status_log_text.setStyleSheet("""
            font-family: "Consolas", "Courier New", monospace;
            font-size: 12px;
            line-height: 1.5;
        """)
        status_log_layout.addWidget(self.status_log_text)
        results_tabs.addTab(status_log_tab, "状态日志")

        right_layout.addWidget(results_tabs)

        # 创建分割器，可调整左右区域比例
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([1000, 600])  # 设置初始宽度比例

        # 设置最小宽度，确保各区域不会被压缩得太小
        left_widget.setMinimumWidth(900)
        right_widget.setMinimumWidth(450)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # 设置默认窗口大小
        self.resize(1600, 900)  # 保持较大的窗口大小

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

            self.total_progress_bar.setValue(
                int(self.completed_combinations / self.total_combinations * 100) if self.total_combinations > 0 else 0)
            self.total_progress_label.setText(
                f"总进度: {self.completed_combinations}/{self.total_combinations} ({self.completed_combinations / self.total_combinations * 100:.1f}%)")

            self.log_message(data.get('message', ''))

        elif update_type == 'info':
            # 普通信息消息
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
            self.batch_progress_label.setText(
                f"当前批次: {self.current_batch}/{self.total_batches} ({batch_progress:.1f}%)")

            self.total_progress_bar.setValue(int(total_progress))
            self.total_progress_label.setText(
                f"总进度: {int(total_progress * self.total_combinations / 100)}/{self.total_combinations} ({total_progress:.1f}%)")

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
            self.log_message(
                f"开始处理参数组合: forward_period={params.get('forward_period')}, generations={params.get('generations')}, population_size={params.get('population_size')}")

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
            factor.get('long_net_returns', 0),
            factor.get('long_valid_trades_ratio', 0),
            factor.get('short_returns', 0),
            factor.get('short_net_returns', 0),
            factor.get('short_valid_trades_ratio', 0)
        ))

        # 按IC值排序
        self.factors_found.sort(key=lambda x: abs(x[2]), reverse=True)

        # 更新表格
        self.results_table.setRowCount(len(self.factors_found))

        for i, (forward_period, expression, ic, stability,
                long_returns, long_net_returns, long_valid_ratio,
                short_returns, short_net_returns, short_valid_ratio) in enumerate(self.factors_found):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(forward_period)))
            self.results_table.setItem(i, 1, QTableWidgetItem(expression))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{ic:.4f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{stability:.4f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{long_returns:.4f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{long_net_returns:.4f}"))
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{long_valid_ratio:.2f}"))
            self.results_table.setItem(i, 7, QTableWidgetItem(f"{short_returns:.4f}"))
            self.results_table.setItem(i, 8, QTableWidgetItem(f"{short_net_returns:.4f}"))
            self.results_table.setItem(i, 9, QTableWidgetItem(f"{short_valid_ratio:.2f}"))

    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # 检查log_text是否存在
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(log_entry)
        else:
            print(f"日志: {log_entry}")  # 如果UI还未初始化，则打印到控制台

        # 检查status_log_text是否存在
        if hasattr(self, 'status_log_text') and self.status_log_text is not None:
            self.status_log_text.append(log_entry)

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

        # 检查组合数标签是否已创建
        if hasattr(self, 'combinations_label'):
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
        factor_type = self.factor_type_combo.currentText()  # 获取选择的因子类型
        ic_threshold = self.ic_threshold_spinbox.value()
        stability_threshold = self.stability_threshold_spinbox.value()
        min_long_return = self.min_long_return_spinbox.value()
        min_short_return = self.min_short_return_spinbox.value()
        enable_segment_test = self.enable_segment_test_checkbox.isChecked()

        # 对test_set_ratio进行检查，使用默认值0.3或从控件获取值
        test_set_ratio = 0.3
        if hasattr(self, 'test_set_ratio_spinbox'):
            test_set_ratio = self.test_set_ratio_spinbox.value()

        max_complexity = self.max_complexity_spinbox.value()
        transaction_fee = self.transaction_fee_spinbox.value()
        min_trade_return = self.min_trade_return_spinbox.value()

        # 将设置保存到配置中
        # 实际应用中应该更新到FIXED_PARAMS中
        self.factor_settings = {
            "factor_type": factor_type,  # 添加因子类型到设置中
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
            f"因子条件设置已应用。选择的因子类型：{factor_type}。这些设置将在下次启动网格搜索时生效。"
        )

        # 记录日志
        self.log_message(
            f"已应用因子条件设置: 因子类型={factor_type}, IC阈值={ic_threshold}, 稳定性阈值={stability_threshold}, " +
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

    def update_factor_type_description(self):
        """更新因子类型描述和推荐参数"""
        # 获取当前选择的因子类型
        factor_type = self.factor_type_combo.currentText()

        # 各种因子类型的描述
        descriptions = {
            "量价获利因子": "量价获利因子结合价格变动和成交量数据，识别市场参与者获利和亏损状态，预测未来价格走势。适合中短期交易。",
            "支撑阻力因子": "支撑阻力因子识别价格的关键支撑和阻力水平，通过反弹和突破模式预测价格行为。适合短期波段交易。",
            "趋势动能因子": "趋势动能因子衡量价格变动趋势的强度和持续性，结合多种技术指标预测趋势延续或反转。适合顺势交易。",
            "波动率因子": "波动率因子基于市场波动性的变化分析未来价格可能的剧烈程度，适合判断市场状态和制定策略选择。",
            "流动性因子": "流动性因子分析交易量模式和买卖压力的变化，识别市场流动性增减的转折点，预测价格可能的大幅变动。"
        }

        # 各种因子类型的推荐参数
        recommended_params = {
            "量价获利因子": {
                "forward_period": [12, 24],
                "population_size": [1000, 3000],
                "generations": [100, 200],
                "tournament_size": [20, 30],
                "function_set": ["add", "sub", "mul", "div", "log", "sqrt", "square"],
                "ic_threshold": 0.05,
                "stability_threshold": 0.3,
                "min_long_return": 0.5,
                "min_short_return": -0.5,
                "transaction_fee": 0.1,
                "min_trade_return": 0.3
            },
            "支撑阻力因子": {
                "forward_period": [6, 12],
                "population_size": [1000, 2000],
                "generations": [100, 150],
                "tournament_size": [20, 30],
                "function_set": ["add", "sub", "mul", "div", "max", "min", "abs", "if_then_else"],
                "ic_threshold": 0.03,
                "stability_threshold": 0.4,
                "min_long_return": 0.3,
                "min_short_return": -0.3,
                "transaction_fee": 0.1,
                "min_trade_return": 0.2
            },
            "趋势动能因子": {
                "forward_period": [12, 24, 36],
                "population_size": [2000, 4000],
                "generations": [150, 250],
                "tournament_size": [30, 40],
                "function_set": ["add", "sub", "mul", "div", "pow", "exp", "sqrt"],
                "ic_threshold": 0.04,
                "stability_threshold": 0.35,
                "min_long_return": 0.6,
                "min_short_return": -0.6,
                "transaction_fee": 0.1,
                "min_trade_return": 0.4
            },
            "波动率因子": {
                "forward_period": [6, 12, 24],
                "population_size": [1000, 2000],
                "generations": [100, 150],
                "tournament_size": [20, 30],
                "function_set": ["add", "sub", "mul", "div", "sqrt", "abs", "square"],
                "ic_threshold": 0.03,
                "stability_threshold": 0.25,
                "min_long_return": 0.4,
                "min_short_return": -0.4,
                "transaction_fee": 0.1,
                "min_trade_return": 0.35
            },
            "流动性因子": {
                "forward_period": [12, 24],
                "population_size": [1000, 3000],
                "generations": [100, 200],
                "tournament_size": [20, 30],
                "function_set": ["add", "sub", "mul", "div", "log", "sqrt"],
                "ic_threshold": 0.04,
                "stability_threshold": 0.3,
                "min_long_return": 0.45,
                "min_short_return": -0.45,
                "transaction_fee": 0.1,
                "min_trade_return": 0.3
            }
        }

        # 更新描述文本
        self.factor_description.setText(descriptions.get(factor_type, "未知因子类型"))

        # 更新函数集显示
        functions = recommended_params.get(factor_type, {}).get("function_set", [])
        function_str = ", ".join(functions)
        self.function_set_label.setText(f"推荐函数集: {function_str}")

        # 更新各项参数设置
        params = recommended_params.get(factor_type, {})

        # 更新第一列的因子条件设置
        if hasattr(self, 'ic_threshold_spinbox'):
            self.ic_threshold_spinbox.setValue(params.get("ic_threshold", 0.05))
        if hasattr(self, 'stability_threshold_spinbox'):
            self.stability_threshold_spinbox.setValue(params.get("stability_threshold", 0.3))
        if hasattr(self, 'min_long_return_spinbox'):
            self.min_long_return_spinbox.setValue(params.get("min_long_return", 0.5))
        if hasattr(self, 'min_short_return_spinbox'):
            self.min_short_return_spinbox.setValue(params.get("min_short_return", -0.5))

        # 更新第二列的因子条件设置
        if hasattr(self, 'transaction_fee_spinbox'):
            self.transaction_fee_spinbox.setValue(params.get("transaction_fee", 0.1))
        if hasattr(self, 'min_trade_return_spinbox'):
            self.min_trade_return_spinbox.setValue(params.get("min_trade_return", 0.3))

        # 特殊处理：仅在选择支撑阻力因子时显示特殊配置区域
        if hasattr(self, 'support_resistance_config_group'):
            self.support_resistance_config_group.setVisible(factor_type == "支撑阻力因子")

        # 记录因子类型变化
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_message(f"已选择因子类型: {factor_type}")

        # 更新因子类型设置
        self.factor_settings["factor_type"] = factor_type

    def apply_sr_config(self):
        """应用支撑阻力因子特殊配置"""
        # 检查当前选择的因子类型
        if self.factor_type_combo.currentText() != "支撑阻力因子":
            QMessageBox.warning(
                self,
                "配置错误",
                "只有在选择支撑阻力因子时才能应用此配置。请先选择支撑阻力因子。"
            )
            return

        # 获取设置值
        detect_bounce = self.bounce_detection_checkbox.isChecked()
        min_bounce_percentage = self.min_bounce_percentage_spinbox.value()
        price_level_importance = self.price_level_importance_spinbox.value()
        volume_confirmation = self.volume_confirmation_checkbox.isChecked()
        pattern_recognition = self.pattern_recognition_checkbox.isChecked()

        # 将设置保存到因子设置中
        sr_special_config = {
            "detect_bounce": detect_bounce,
            "min_bounce_percentage": min_bounce_percentage,
            "price_level_importance": price_level_importance,
            "volume_confirmation": volume_confirmation,
            "pattern_recognition": pattern_recognition
        }

        # 更新因子设置
        self.factor_settings.update(sr_special_config)

        # 更新固定参数，确保在搜索时应用这些特殊设置
        special_sr_params = {
            "detect_bounce": detect_bounce,
            "min_bounce_percentage": min_bounce_percentage,
            "price_level_importance": price_level_importance,
            "volume_confirmation": volume_confirmation,
            "pattern_recognition": pattern_recognition
        }
        self.fixed_params.update(special_sr_params)

        # 显示确认消息
        config_summary = f"支撑阻力因子特殊配置已应用:\n" + \
                         f"- 反弹检测: {'启用' if detect_bounce else '禁用'}\n" + \
                         f"- 最小反弹幅度: {min_bounce_percentage}%\n" + \
                         f"- 价格水平重要性: {price_level_importance}\n" + \
                         f"- 成交量确认: {'启用' if volume_confirmation else '禁用'}\n" + \
                         f"- 价格形态识别: {'启用' if pattern_recognition else '禁用'}\n"

        QMessageBox.information(
            self,
            "支撑阻力配置已应用",
            config_summary
        )

        # 记录日志
        self.log_message("已应用支撑阻力因子特殊配置:")
        self.log_message(f"反弹检测: {'启用' if detect_bounce else '禁用'}")
        self.log_message(f"最小反弹幅度: {min_bounce_percentage}%")
        self.log_message(f"价格水平重要性: {price_level_importance}")
        self.log_message(f"成交量确认: {'启用' if volume_confirmation else '禁用'}")
        self.log_message(f"价格形态识别: {'启用' if pattern_recognition else '禁用'}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格，更接近Windows 11
    ui = GridSearchUI()
    sys.exit(app.exec()) 