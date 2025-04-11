"""
因子策略UI模块

提供因子策略的展示、参数调整和回测控制功能
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入PyQt6
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QPushButton, QGroupBox, QSplitter, 
                           QFrame, QGridLayout, QSizePolicy, QApplication,
                           QTabWidget, QTextEdit, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QSlider, QFormLayout, QLineEdit, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThread

# 导入项目相关模块
from factor_trading_strategy import FactorTradingStrategy

class BacktestWorker(QThread):
    """回测工作线程"""
    # 定义信号
    progress_signal = pyqtSignal(int)  # 进度信号
    status_signal = pyqtSignal(str)  # 状态信号
    finished_signal = pyqtSignal(object)  # 完成信号，传递结果数据
    error_signal = pyqtSignal(str)  # 错误信号
    
    def __init__(self, strategy, data):
        super().__init__()
        self.strategy = strategy
        self.data = data
        self.total_rows = len(data)
        # 重定向标准输出
        self.old_stdout = sys.stdout
        self.captured_output = []
        
    def run(self):
        try:
            self.status_signal.emit("开始回测...")
            
            # 捕获print输出
            class StdoutRedirector:
                def __init__(self, worker):
                    self.worker = worker
                
                def write(self, text):
                    if text.strip() and not text.startswith('\r'):
                        self.worker.captured_output.append(text)
                        self.worker.status_signal.emit(text.strip())
                        
                        # 检查是否是进度更新
                        if "回测进度:" in text:
                            try:
                                parts = text.split()
                                current, total = parts[1].split('/')
                                progress = int((int(current) / int(total)) * 100)
                                self.worker.progress_signal.emit(progress)
                            except:
                                pass
                
                def flush(self):
                    pass
            
            # 重定向stdout
            sys.stdout = StdoutRedirector(self)
            
            # 执行回测
            result = self.strategy.backtest(self.data)
            
            # 恢复标准输出
            sys.stdout = self.old_stdout
            
            # 发送完成信号
            self.finished_signal.emit(result)
            self.status_signal.emit("回测完成！")
        except Exception as e:
            # 恢复标准输出
            sys.stdout = self.old_stdout
            self.error_signal.emit(f"回测出错: {str(e)}")
            print(f"回测出错: {str(e)}")  # 打印错误信息到控制台

class FactorStrategyUI(QWidget):
    """因子策略UI组件"""
    
    # 定义信号，当回测数据准备好时，通知K线图界面显示
    backtest_result_ready = pyqtSignal(object, object)  # 传递 strategy 和 result
    # 定义信号，请求获取K线图数据
    request_kline_data = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.factor_expression = "div(X10, log(X55))"  # 默认因子表达式
        self.strategy = None
        self.backtest_result = None
        self.data = None
        self.init_ui()
        self.load_saved_params()  # 加载保存的参数
        
    def init_ui(self):
        """初始化UI界面"""
        main_layout = QVBoxLayout()
        
        # 创建因子信息展示区
        info_group = QGroupBox("因子信息")
        info_layout = QVBoxLayout()
        
        # 因子表达式
        expression_layout = QHBoxLayout()
        expression_layout.addWidget(QLabel("因子表达式:"))
        self.expression_edit = QLineEdit(self.factor_expression)
        self.expression_edit.setMinimumWidth(300)
        expression_layout.addWidget(self.expression_edit)
        info_layout.addLayout(expression_layout)
        
        # 因子描述
        self.factor_desc = QTextEdit()
        self.factor_desc.setReadOnly(True)
        self.factor_desc.setMinimumHeight(100)
        self.factor_desc.setText("该因子使用成交量变化率除以价格波动与成交量比率的对数值。\n\n"
                               "主要捕捉市场中成交量和价格之间的关系变化，当成交量上升而价格波动较小时，"
                               "可能预示着更强的趋势动量。反之，如果成交量下降而价格波动较大，可能表明市场正在筋疲力尽。")
        info_layout.addWidget(self.factor_desc)
        
        # 策略原理
        strategy_desc = QTextEdit()
        strategy_desc.setReadOnly(True)
        strategy_desc.setMinimumHeight(100)
        strategy_desc.setText("策略原理:\n\n"
                            "1. 计算因子值: div(X10, log(X55)) - 成交量变化率除以价格波动率的对数\n"
                            "2. 排序方法: 使用100周期滚动窗口，计算因子值的百分位排名\n"
                            "3. 信号生成: 当因子排名>80%时做多，<20%时做空\n"
                            "4. 交易执行: 考虑交易成本和最小收益要求后开仓，反向信号或无信号时平仓")
        info_layout.addWidget(strategy_desc)
        
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)
        
        # 创建参数设置区
        params_group = QGroupBox("策略参数")
        params_layout = QGridLayout()
        
        # 预测周期
        params_layout.addWidget(QLabel("预测周期(K线数量):"), 0, 0)
        self.forward_period_spin = QSpinBox()
        self.forward_period_spin.setRange(1, 100)
        self.forward_period_spin.setValue(48)
        params_layout.addWidget(self.forward_period_spin, 0, 1)
        
        # 交易手续费率
        params_layout.addWidget(QLabel("交易手续费率(%):"), 0, 2)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0, 1)
        self.fee_spin.setSingleStep(0.01)
        self.fee_spin.setValue(0.04)
        params_layout.addWidget(self.fee_spin, 0, 3)
        
        # 最小交易收益
        params_layout.addWidget(QLabel("最小交易收益(%):"), 1, 0)
        self.min_return_spin = QDoubleSpinBox()
        self.min_return_spin.setRange(0, 1)
        self.min_return_spin.setSingleStep(0.01)
        self.min_return_spin.setValue(0.05)
        params_layout.addWidget(self.min_return_spin, 1, 1)
        
        # 止损比例
        params_layout.addWidget(QLabel("止损比例(%):"), 1, 2)
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.1, 10)
        self.stop_loss_spin.setSingleStep(0.1)
        self.stop_loss_spin.setValue(0.5)
        params_layout.addWidget(self.stop_loss_spin, 1, 3)
        
        # 初始资金
        params_layout.addWidget(QLabel("初始资金(USDT):"), 2, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 100000)
        self.capital_spin.setSingleStep(100)
        self.capital_spin.setValue(1000)
        params_layout.addWidget(self.capital_spin, 2, 1)
        
        # 做多阈值
        params_layout.addWidget(QLabel("做多阈值(百分位):"), 3, 0)
        self.long_threshold_spin = QDoubleSpinBox()
        self.long_threshold_spin.setRange(0.5, 1.0)
        self.long_threshold_spin.setSingleStep(0.01)
        self.long_threshold_spin.setValue(0.8)
        params_layout.addWidget(self.long_threshold_spin, 3, 1)
        
        # 做空阈值
        params_layout.addWidget(QLabel("做空阈值(百分位):"), 3, 2)
        self.short_threshold_spin = QDoubleSpinBox()
        self.short_threshold_spin.setRange(0, 0.5)
        self.short_threshold_spin.setSingleStep(0.01)
        self.short_threshold_spin.setValue(0.2)
        params_layout.addWidget(self.short_threshold_spin, 3, 3)
        
        # 添加交易方向选择
        params_layout.addWidget(QLabel("交易方向:"), 4, 0)
        self.trade_direction_combo = QComboBox()
        self.trade_direction_combo.addItems(["只做多", "只做空", "多空均做"])
        self.trade_direction_combo.setCurrentIndex(0)  # 默认"只做多"
        params_layout.addWidget(self.trade_direction_combo, 4, 1)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # 创建操作区
        control_layout = QHBoxLayout()
        
        # 保存参数按钮
        self.save_params_btn = QPushButton("保存参数")
        self.save_params_btn.clicked.connect(self.save_strategy_params)
        control_layout.addWidget(self.save_params_btn)
        
        # 加载数据按钮改为从K线图获取数据
        self.load_data_btn = QPushButton("从K线图获取数据")
        self.load_data_btn.clicked.connect(self.load_data_from_kline_view)
        control_layout.addWidget(self.load_data_btn)
        
        # 回测按钮
        self.backtest_btn = QPushButton("开始回测")
        self.backtest_btn.clicked.connect(self.start_backtest)
        self.backtest_btn.setEnabled(False)  # 初始禁用，等待数据加载
        control_layout.addWidget(self.backtest_btn)
        
        # 查看结果按钮
        self.view_results_btn = QPushButton("查看结果")
        self.view_results_btn.clicked.connect(self.view_results)
        self.view_results_btn.setEnabled(False)  # 初始禁用，等待回测完成
        control_layout.addWidget(self.view_results_btn)
        
        # 保存结果按钮
        self.save_results_btn = QPushButton("保存结果")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)  # 初始禁用，等待回测完成
        control_layout.addWidget(self.save_results_btn)
        
        main_layout.addLayout(control_layout)
        
        # 添加进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("回测进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)
        
        # 状态栏
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("状态:"))
        self.status_label = QLabel("就绪")
        self.status_label.setMinimumWidth(300)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        # 数据信息
        self.data_info_label = QLabel("未加载数据")
        status_layout.addWidget(self.data_info_label)
        
        main_layout.addLayout(status_layout)
        
        self.setLayout(main_layout)
    
    def load_data_from_kline_view(self):
        """从K线图视图获取数据"""
        self.status_label.setText("正在从K线图获取数据...")
        
        # 发射信号请求获取K线数据，主窗口中的槽函数将处理此请求
        self.request_kline_data.emit()
        
        # 信号处理函数将返回数据，但由于是异步的，我们需要另一个槽函数来接收数据
        # 主窗口将调用set_kline_data方法设置数据
        
    def set_kline_data(self, data):
        """接收从K线图视图获取的数据
        
        参数:
            data (pd.DataFrame): K线数据
        """
        if data is None or data.empty:
            self.status_label.setText("未能从K线图获取数据，请先在K线图标签页加载数据")
            return
            
        self.data = data
        
        # 更新UI状态
        self.data_info_label.setText(f"数据: {len(data)}行 ({data.index[0]} 到 {data.index[-1]})")
        self.status_label.setText(f"成功获取全量K线数据，共 {len(data)} 条记录")
        self.backtest_btn.setEnabled(True)
    
    def load_data(self):
        """原始的数据加载方法，保留但不再使用"""
        self.status_label.setText("不再使用此方法加载数据，请使用'从K线图获取数据'按钮")
        
    def save_strategy_params(self):
        """保存策略参数设置"""
        # 从UI获取参数
        factor_expression = self.expression_edit.text()
        forward_period = self.forward_period_spin.value()
        transaction_fee = self.fee_spin.value()
        min_trade_return = self.min_return_spin.value()
        stop_loss = self.stop_loss_spin.value()
        initial_capital = self.capital_spin.value()
        long_threshold = self.long_threshold_spin.value()
        short_threshold = self.short_threshold_spin.value()
        trade_direction = self.trade_direction_combo.currentText()
        
        # 创建参数摘要
        params_summary = f"""
策略参数设置已保存:

因子表达式: {factor_expression}
预测周期: {forward_period} 根K线
交易手续费率: {transaction_fee}%
最小交易收益(止盈): {min_trade_return}%
止损比例: {stop_loss}%
初始资金: {initial_capital} USDT
做多阈值: {long_threshold*100:.1f}%
做空阈值: {short_threshold*100:.1f}%
交易方向: {trade_direction}
        """
        
        # 将参数保存到配置文件
        try:
            import json
            import os
            
            config = {
                "factor_expression": factor_expression,
                "forward_period": forward_period,
                "transaction_fee": transaction_fee,
                "min_trade_return": min_trade_return,
                "stop_loss": stop_loss,
                "initial_capital": initial_capital,
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "trade_direction": trade_direction
            }
            
            # 确保目录存在
            os.makedirs('config', exist_ok=True)
            
            # 保存到JSON文件
            with open('config/strategy_params.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            # 显示成功消息
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "参数保存成功", params_summary)
            
            self.status_label.setText("策略参数已保存")
        except Exception as e:
            self.status_label.setText(f"保存参数时出错: {str(e)}")
    
    def start_backtest(self):
        """开始回测"""
        if self.data is None:
            self.status_label.setText("请先加载数据")
            return
        
        # 获取策略参数
        self.factor_expression = self.expression_edit.text()
        forward_period = self.forward_period_spin.value()
        transaction_fee = self.fee_spin.value()
        min_trade_return = self.min_return_spin.value()
        stop_loss = self.stop_loss_spin.value()  # 获取止损参数
        initial_capital = self.capital_spin.value()
        long_threshold = self.long_threshold_spin.value()
        short_threshold = self.short_threshold_spin.value()
        trade_direction = self.trade_direction_combo.currentText()
        
        # 当数据量较大时提示用户
        if len(self.data) > 50000:
            from PyQt6.QtWidgets import QMessageBox
            response = QMessageBox.question(
                self,
                "大数据量警告",
                f"您将使用 {len(self.data)} 条K线记录进行回测，这可能会消耗较多内存并需要较长时间。\n"
                f"是否继续？\n\n"
                f"提示：回测期间请勿操作其他功能，等待回测完成。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if response == QMessageBox.StandardButton.No:
                return
        
        # 创建策略实例
        self.strategy = FactorTradingStrategy(
            initial_capital=initial_capital,
            transaction_fee_rate=transaction_fee,
            min_trade_return=min_trade_return,
            stop_loss=stop_loss,  # 添加止损参数
            forward_period=forward_period,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            trade_direction=trade_direction,
            factor_expression=self.factor_expression
        )
        
        # 创建并启动回测线程
        self.backtest_worker = BacktestWorker(self.strategy, self.data)
        self.backtest_worker.status_signal.connect(self.update_status)
        self.backtest_worker.progress_signal.connect(self.handle_progress)
        self.backtest_worker.error_signal.connect(self.handle_error)
        self.backtest_worker.finished_signal.connect(self.handle_backtest_finished)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 禁用所有控制按钮
        self.backtest_btn.setEnabled(False)
        self.load_data_btn.setEnabled(False)
        self.view_results_btn.setEnabled(False)
        self.save_results_btn.setEnabled(False)
        
        self.status_label.setText(f"正在回测 {len(self.data)} 条K线数据，请耐心等待...")
        self.backtest_worker.start()
    
    def update_status(self, status):
        """更新状态标签"""
        self.status_label.setText(status)
    
    def handle_progress(self, progress):
        """处理进度更新"""
        self.progress_bar.setValue(progress)
    
    def handle_error(self, error_msg):
        """处理错误"""
        self.status_label.setText(error_msg)
        self.backtest_btn.setEnabled(True)
    
    def handle_backtest_finished(self, result):
        """处理回测完成"""
        self.backtest_result = result
        
        # 恢复按钮状态
        self.backtest_btn.setEnabled(True)
        self.load_data_btn.setEnabled(True)
        self.view_results_btn.setEnabled(True)
        self.save_results_btn.setEnabled(True)
        
        # 设置进度条为100%完成
        self.progress_bar.setValue(100)
        
        # 输出策略摘要
        try:
            # 尝试打印英文摘要
            self.strategy.print_english_summary()
        except Exception as e:
            self.status_label.setText(f"打印摘要时出错: {str(e)}")
        
        # 显示总交易次数和收益率
        if hasattr(self.strategy, 'metrics') and self.strategy.metrics:
            trade_count = self.strategy.metrics.get('trade_count', 0)
            total_return = self.strategy.metrics.get('total_return', 0)
            win_rate = self.strategy.metrics.get('win_rate', 0)
            
            self.status_label.setText(
                f"回测完成: {trade_count}笔交易, 胜率{win_rate:.2f}%, 总收益{total_return:.2f}%"
            )
        else:
            self.status_label.setText("回测完成，但未生成完整的统计数据")
        
        # 发射信号通知K线图界面
        self.backtest_result_ready.emit(self.strategy, result)
    
    def view_results(self):
        """查看回测结果"""
        if self.strategy is None or self.backtest_result is None:
            self.status_label.setText("没有回测结果可查看")
            return
        
        try:
            # 创建结果展示窗口
            from PyQt6.QtWidgets import QDialog, QTabWidget
            dialog = QDialog(self)
            dialog.setWindowTitle("回测结果")
            dialog.resize(800, 600)
            
            layout = QVBoxLayout()
            tabs = QTabWidget()
            
            # 添加资金曲线图表
            equity_tab = QWidget()
            equity_layout = QVBoxLayout()
            figure = Figure(figsize=(8, 6), dpi=100)
            canvas = FigureCanvas(figure)
            equity_layout.addWidget(canvas)
            equity_tab.setLayout(equity_layout)
            tabs.addTab(equity_tab, "资金曲线")
            
            # 绘制资金曲线
            ax = figure.add_subplot(111)
            ax.plot(self.strategy.equity_curve, label='资金曲线')
            ax.set_title('回测资金曲线')
            ax.set_xlabel('时间')
            ax.set_ylabel('资金(USDT)')
            ax.grid(True)
            ax.legend()
            canvas.draw()
            
            # 添加交易记录表格
            from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
            trades_tab = QWidget()
            trades_layout = QVBoxLayout()
            trades_table = QTableWidget()
            trades_layout.addWidget(trades_table)
            trades_tab.setLayout(trades_layout)
            tabs.addTab(trades_tab, "交易记录")
            
            # 填充交易记录
            if self.strategy.trades:
                trades_table.setColumnCount(11)
                trades_table.setRowCount(len(self.strategy.trades))
                trades_table.setHorizontalHeaderLabels(
                    ["入场时间", "入场价格", "出场时间", "出场价格", "方向", "数量", 
                     "毛利润", "手续费", "净利润", "收益率", "持仓周期"]
                )
                
                for i, trade in enumerate(self.strategy.trades):
                    trades_table.setItem(i, 0, QTableWidgetItem(str(trade['entry_time'])))
                    trades_table.setItem(i, 1, QTableWidgetItem(f"{trade['entry_price']:.2f}"))
                    trades_table.setItem(i, 2, QTableWidgetItem(str(trade['exit_time'])))
                    trades_table.setItem(i, 3, QTableWidgetItem(f"{trade['exit_price']:.2f}"))
                    direction = "多" if trade['position'] > 0 else "空"
                    trades_table.setItem(i, 4, QTableWidgetItem(direction))
                    trades_table.setItem(i, 5, QTableWidgetItem(f"{trade['shares']:.4f}"))
                    trades_table.setItem(i, 6, QTableWidgetItem(f"{trade['pnl']:.2f}"))
                    trades_table.setItem(i, 7, QTableWidgetItem(f"{trade['fee']:.2f}"))
                    trades_table.setItem(i, 8, QTableWidgetItem(f"{trade['net_pnl']:.2f}"))
                    trades_table.setItem(i, 9, QTableWidgetItem(f"{trade['return']*100:.2f}%"))
                    trades_table.setItem(i, 10, QTableWidgetItem(f"{trade.get('holding_periods', '未知')}"))
            
            # 添加统计指标
            stats_tab = QWidget()
            stats_layout = QVBoxLayout()
            stats_text = QTextEdit()
            stats_text.setReadOnly(True)
            stats_layout.addWidget(stats_text)
            stats_tab.setLayout(stats_layout)
            tabs.addTab(stats_tab, "统计指标")
            
            # 填充统计指标
            stats_html = f"""
            <h2>回测统计指标</h2>
            <table border="0" cellspacing="5" cellpadding="5" width="100%">
                <tr>
                    <td><b>初始资金:</b></td>
                    <td>{self.strategy.initial_capital:.2f} USDT</td>
                    <td><b>最终资金:</b></td>
                    <td>{self.strategy.equity_curve.iloc[-1]:.2f} USDT</td>
                </tr>
                <tr>
                    <td><b>总收益率:</b></td>
                    <td>{self.strategy.metrics['total_return']:.2f}%</td>
                    <td><b>年化收益:</b></td>
                    <td>{self.strategy.metrics['annual_return']:.2f}%</td>
                </tr>
                <tr>
                    <td><b>最大回撤:</b></td>
                    <td>{self.strategy.metrics['max_drawdown']:.2f}%</td>
                    <td><b>交易次数:</b></td>
                    <td>{self.strategy.metrics['trade_count']}</td>
                </tr>
                <tr>
                    <td><b>胜率:</b></td>
                    <td>{self.strategy.metrics['win_rate']:.2f}%</td>
                    <td><b>平均收益:</b></td>
                    <td>{self.strategy.metrics['avg_return']:.2f}%</td>
                </tr>
                <tr>
                    <td><b>平均盈利:</b></td>
                    <td>{self.strategy.metrics['avg_win']:.2f}%</td>
                    <td><b>平均亏损:</b></td>
                    <td>{self.strategy.metrics['avg_loss']:.2f}%</td>
                </tr>
                <tr>
                    <td><b>盈亏比:</b></td>
                    <td>{self.strategy.metrics['profit_loss_ratio']:.2f}</td>
                    <td><b>夏普比率:</b></td>
                    <td>{self.strategy.metrics['sharpe_ratio']:.2f}</td>
                </tr>
                <tr>
                    <td><b>平均持仓周期:</b></td>
                    <td>{self.strategy.metrics.get('avg_holding_periods', 0):.2f}根K线</td>
                    <td><b>交易方向:</b></td>
                    <td>{self.strategy.trade_direction}</td>
                </tr>
            </table>
            
            <h3>因子参数</h3>
            <table border="0" cellspacing="5" cellpadding="5" width="100%">
                <tr>
                    <td><b>因子表达式:</b></td>
                    <td colspan="3">{self.factor_expression}</td>
                </tr>
                <tr>
                    <td><b>预测周期:</b></td>
                    <td>{self.strategy.forward_period} 根K线</td>
                    <td><b>交易手续费率:</b></td>
                    <td>{self.strategy.transaction_fee_rate*100:.4f}%</td>
                </tr>
                <tr>
                    <td><b>最小交易收益:</b></td>
                    <td>{self.strategy.min_trade_return*100:.4f}%</td>
                    <td><b>止损比例:</b></td>
                    <td>{self.strategy.stop_loss*100:.4f}%</td>
                </tr>
                <tr>
                    <td><b>做多阈值:</b></td>
                    <td>{self.strategy.long_threshold*100:.1f}%</td>
                    <td><b>做空阈值:</b></td>
                    <td>{self.strategy.short_threshold*100:.1f}%</td>
                </tr>
                <tr>
                    <td><b>交易方向:</b></td>
                    <td>{self.strategy.trade_direction}</td>
                    <td><b>预测周期:</b></td>
                    <td>{self.strategy.forward_period} 根K线</td>
                </tr>
            </table>
            """
            stats_text.setHtml(stats_html)
            
            layout.addWidget(tabs)
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            self.status_label.setText(f"查看结果时出错: {str(e)}")
    
    def save_results(self):
        """保存回测结果"""
        if self.strategy is None or self.backtest_result is None:
            self.status_label.setText("没有回测结果可保存")
            return
        
        try:
            # 保存结果
            name = f"factor_{self.factor_expression.replace('(', '').replace(')', '').replace(',', '_')}"
            self.strategy.save_results(name)
            self.status_label.setText(f"结果已保存到 trading_results 目录")
        except Exception as e:
            self.status_label.setText(f"保存结果时出错: {str(e)}")
    
    def load_saved_params(self):
        """加载保存的策略参数"""
        try:
            import json
            import os
            
            config_file = 'config/strategy_params.json'
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 设置UI控件值
                self.expression_edit.setText(config.get('factor_expression', self.factor_expression))
                self.forward_period_spin.setValue(config.get('forward_period', 48))
                self.fee_spin.setValue(config.get('transaction_fee', 0.04))
                self.min_return_spin.setValue(config.get('min_trade_return', 0.05))
                self.stop_loss_spin.setValue(config.get('stop_loss', 0.5))
                self.capital_spin.setValue(config.get('initial_capital', 1000.0))
                self.long_threshold_spin.setValue(config.get('long_threshold', 0.8))
                self.short_threshold_spin.setValue(config.get('short_threshold', 0.2))
                
                # 设置交易方向
                direction = config.get('trade_direction', "只做多")
                index = self.trade_direction_combo.findText(direction)
                if index >= 0:
                    self.trade_direction_combo.setCurrentIndex(index)
                
                self.status_label.setText("已加载保存的策略参数")
        except Exception as e:
            self.status_label.setText(f"加载参数时出错: {str(e)}")

# 集成到现有UI系统
class MainWindow(QTabWidget):
    """主窗口，集成因子策略UI到标签页系统"""
    
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 设置窗口标题和大小
        self.setWindowTitle("ETH 交易分析系统")
        self.resize(1200, 800)
        
        # 添加因子策略UI标签页
        self.factor_strategy_tab = FactorStrategyUI()
        self.addTab(self.factor_strategy_tab, "因子策略")
        
        # 这里可以添加其他标签页，如K线图页面
        # 简单创建一个占位符K线图页面
        self.kline_tab = QWidget()
        kline_layout = QVBoxLayout()
        kline_layout.addWidget(QLabel("K线图和交易信号将在此显示"))
        self.kline_tab.setLayout(kline_layout)
        self.addTab(self.kline_tab, "K线图")
        
        # 关联回测结果信号到显示函数
        self.factor_strategy_tab.backtest_result_ready.connect(self.show_backtest_on_kline)
    
    def show_backtest_on_kline(self, strategy, result):
        """在K线图上显示回测结果"""
        # 这里需要根据您的K线图UI组件来进行集成
        # 假设您有一个名为 kline_view 的K线图组件，可以通过它的 show_trades 方法显示交易
        
        # 切换到K线图标签页
        self.setCurrentWidget(self.kline_tab)
        
        # 提示用户
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "回测完成", "回测已完成！\n\n您可以在K线图页面查看交易情况。")

# 如果独立运行，则显示UI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 