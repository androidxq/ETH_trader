"""
强化学习策略UI模块
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import math

# 导入matplotlib相关组件
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox,
    QTabWidget, QTextEdit, QProgressBar, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog, QApplication,
    QFormLayout, QGridLayout, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor

# 导入强化学习组件
from rl_strategies.trainer import RLTrainer
from rl_strategies.rl_training_thread import RLTrainingThread

class RLStrategiesUI(QWidget):
    """强化学习策略UI类"""

    # 定义信号
    data_updated_signal = pyqtSignal()  # 数据更新信号

    def __init__(self, parent=None):
        """
        初始化UI

        参数:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 设置窗口标题和大小
        self.setWindowTitle("强化学习策略训练器")
        self.resize(1280, 800)

        # 数据相关变量
        self.kline_data = None
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.stop_requested = False
        self.current_episode = 0
        self.new_episode_started = False
        self.episode_start_time = time.time()
        self.rewards_history = []
        self.returns_history = []
        self.learning_rates_history = []
        self.learning_rate_steps = []
        self.portfolio_values = []
        self.best_model = None
        self.accumulated_rewards = []
        self.last_plots_update_time = 0
        self.min_update_interval = 0.5  # 更新图表的最小时间间隔（秒）
        self.max_history_size = 5000  # 历史数据的最大长度
        self.enable_data_compression = True  # 是否启用数据压缩
        self.compress_threshold = 1000  # 压缩阈值
        self.training_trades = []
        self.evaluation_trades = []
        self.returns_steps = []
        self.rewards_steps = []
        
        # 用于跟踪训练步数的计数器
        self.ui_step_counter = 0
        
        # 用于跟踪探索率的变量
        self.ui_epsilon_value = 1.0  # 初始探索率为1.0
        
        # 探索率更新定时器
        self.epsilon_timer = QTimer(self)
        self.epsilon_timer.setInterval(2000)  # 每2秒更新一次
        self.epsilon_timer.timeout.connect(self.update_epsilon_from_agent)
        
        # 初始化UI
        self.matplotlib_available = False  # 默认不可用
        self.init_ui()
        
        # 尝试初始化图表
        self.init_plots()
        
        # 创建定时更新UI的定时器
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(500)  # 500毫秒间隔
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start()
        
        # 设置日志最大行数
        if hasattr(self, 'log_text'):
            self.log_text.document().setMaximumBlockCount(500)  # 限制最大显示500行

    def _data_changed(self, new_data):
        """检查数据是否发生实质性变化"""
        if not new_data:
            return False

        # 计算数据的哈希值
        import hashlib
        data_str = ','.join(map(str, new_data))
        current_hash = hashlib.md5(data_str.encode()).hexdigest()

        # 检查是否与上次数据相同
        changed = current_hash != self._last_lr_data_hash
        self._last_lr_data_hash = current_hash
        return changed

    def cleanup_data(self):
        """清理过期数据"""
        # 检查并清理学习率历史数据
        if hasattr(self, 'learning_rates_history') and len(self.learning_rates_history) > self.max_history_size:
            if self.enable_data_compression and len(self.learning_rates_history) > self.compress_threshold:
                # 使用数据压缩
                self.learning_rates_history, self.learning_rate_steps = self.compress_data(
                    self.learning_rates_history, self.learning_rate_steps)
            else:
                # 简单截断，只保留最新数据
                self.learning_rates_history = self.learning_rates_history[-self.max_history_size:]
                if hasattr(self, 'learning_rate_steps'):
                    self.learning_rate_steps = self.learning_rate_steps[-self.max_history_size:]

            print(f"清理学习率历史数据: 当前数据点数={len(self.learning_rates_history)}")

        # 检查并清理奖励历史数据
        if hasattr(self, 'accumulated_rewards') and len(self.accumulated_rewards) > self.max_history_size:
            self.accumulated_rewards = self.accumulated_rewards[-self.max_history_size:]

        # 清理回报率历史数据
        if hasattr(self, 'returns_history') and len(self.returns_history) > self.max_history_size:
            self.returns_history = self.returns_history[-self.max_history_size:]

    def _reset_history_data(self):
        """重置历史数据"""
        # 完全清空历史数据，而不是保留之前的数据
        self.new_episode_started = True
        self.episode_start_time = time.time()

        # 清空所有历史数据
        self.rewards_history = []
        self.returns_history = []
        self.learning_rates_history = []
        self.learning_rate_steps = []
        self.portfolio_values = []
        self.returns_steps = []
        self.rewards_steps = []
        self.current_episode_returns = None

        # 将更新时间戳重置，确保下一次更新会立即执行
        self.last_plots_update_time = 0

        # 重置图表显示
        if hasattr(self, 'rewards_ax') and self.matplotlib_available:
            self.rewards_ax.clear()
            self.rewards_ax.set_title('训练奖励曲线')
            self.rewards_ax.set_xlabel('训练步数')
            self.rewards_ax.set_ylabel('累积奖励')
            self.rewards_ax.grid(True)
            self.rewards_figure.tight_layout()
            self.rewards_canvas.draw()

        # 如果有返回历史图表，也重置它
        if hasattr(self, 'returns_plot'):
            try:
                self.returns_plot.clear()
            except Exception as e:
                print(f"ERROR: 重置收益率曲线图时出错: {str(e)}")
                
        # 重置学习率曲线图
        if hasattr(self, 'learning_rate_plot'):
            try:
                self.learning_rate_plot.clear()
                print("已重置学习率曲线图")
            except Exception as e:
                print(f"ERROR: 重置学习率曲线图时出错: {str(e)}")

        # 注意：不要重置探索率文本，以便用户可以查看历史探索率记录
        # if hasattr(self, 'epsilon_text'):
        #     self.epsilon_text.clear()
        #     print("已重置探索率显示文本")

        print("DEBUG: 历史数据和图表已完全重置，保留探索率历史记录")

    def update_ui(self):
        """更新UI显示"""
        if not self.is_training:
            return

        # 更新训练进度和状态
        if self.training_thread and self.training_thread.isRunning():
            # 更新训练数据
            if hasattr(self.trainer, 'get_latest_metrics'):
                latest_metrics = self.trainer.get_latest_metrics()
                if latest_metrics:
                    # 检查是否是新回合开始
                    current_episode = latest_metrics.get('episode')
                    previous_episode = getattr(self, 'current_episode', None)
                    if current_episode != previous_episode:
                        # 更新当前回合号，但不清空历史数据
                        self.current_episode = current_episode
                        # 记录新回合开始的标记
                        self.new_episode_started = True
                        # 记录回合开始时间
                        self.episode_start_time = time.time()

                    # 更新历史数据
                    if 'reward' in latest_metrics:
                        if not hasattr(self, 'rewards_history'):
                            self.rewards_history = []
                        self.rewards_history.append(latest_metrics['reward'])
                    if 'return' in latest_metrics:
                        if not hasattr(self, 'returns_history'):
                            self.returns_history = []
                        self.returns_history.append(latest_metrics['return'])
                    # 添加更详细的学习率数据收集调试信息
                    if 'learning_rate' in latest_metrics:
                        if not hasattr(self, 'learning_rates_history'):
                            self.learning_rates_history = []
                            print("DEBUG: 初始化学习率历史数组")
                        
                        current_lr = latest_metrics['learning_rate']
                        self.learning_rates_history.append(current_lr)
                        
                        # 如果没有步数数据，创建它
                        if not hasattr(self, 'learning_rate_steps'):
                            self.learning_rate_steps = []
                        
                        # 添加步数数据
                        next_step = self.learning_rate_steps[-1] + 1 if self.learning_rate_steps else 1
                        self.learning_rate_steps.append(next_step)
                        
                        print(f"DEBUG: 添加学习率数据点: {current_lr:.6f}, 步数: {next_step}, 总数据点: {len(self.learning_rates_history)}")
                    if 'portfolio_value' in latest_metrics:
                        if not hasattr(self, 'portfolio_values'):
                            self.portfolio_values = []
                        self.portfolio_values.append(latest_metrics['portfolio_value'])

                    # 更新图表
                    self.update_plots()

                    # 更新状态标签
                    if 'episode' in latest_metrics:
                        self.status_label.setText(f"正在训练: 第{latest_metrics['episode']}轮")

                    # 更新进度条
                    if 'progress' in latest_metrics:
                        self.progress_bar.setValue(int(latest_metrics['progress'] * 100))

    def update_plots(self):
        """更新所有图表"""
        # 标记本轮更新时间
        self.last_plots_update_time = time.time()

        # 更新奖励曲线
        if hasattr(self, 'rewards_history') and self.rewards_history:
            try:
                self.update_rewards_plot(self.rewards_history)
            except Exception as e:
                print(f"ERROR: 更新奖励曲线时出错: {str(e)}")

        # 更新学习率曲线，增加更多调试信息
        if hasattr(self, 'learning_rates_history') and self.learning_rates_history:
            try:
                print(f"DEBUG: 尝试更新学习率曲线，数据点数: {len(self.learning_rates_history)}")
                self.update_learning_rate_plot(self.learning_rates_history)
            except Exception as e:
                print(f"ERROR: 更新学习率曲线时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
        else:
            print("DEBUG: 没有学习率数据，跳过学习率曲线更新")

        # 更新收益曲线
        if hasattr(self, 'returns_history') and self.returns_history:
            try:
                self.update_returns_plot(self.returns_history)
            except Exception as e:
                print(f"ERROR: 更新收益曲线时出错: {str(e)}")

        # 触发Qt事件处理，确保UI及时更新
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()


    def init_ui(self):
        """初始化UI元素"""
        # 创建主布局
        main_layout = QVBoxLayout(self)

        # 创建分割器
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 初始化图表显示控制变量
        self.show_rewards_plot = True
        self.show_returns_plot = True
        self.show_learning_rate_plot = True
        self.show_portfolio_plot = True

        # 左侧配置面板
        self.config_panel = QWidget()
        config_layout = QVBoxLayout(self.config_panel)

        # 创建标签页控件
        self.config_tabs = QTabWidget()



        # ==================== 模型选择标签页 ====================
        self.model_config_tab = QWidget()
        model_layout = QVBoxLayout(self.model_config_tab)

        # 模型选择
        model_group = QGroupBox("模型选择")
        model_select_layout = QVBoxLayout()

        # 模型类型选择
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("选择模型类型:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["DQN", "PPO", "A2C", "DDPG"])
        self.model_type_combo.currentIndexChanged.connect(self.on_model_changed)
        model_type_layout.addWidget(self.model_type_combo)
        model_select_layout.addLayout(model_type_layout)

        # 添加选项
        self.double_dqn_check = QCheckBox("使用Double DQN")
        self.double_dqn_check.setChecked(True)
        model_select_layout.addWidget(self.double_dqn_check)

        # 模型隐藏层
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("隐藏层大小:"))
        self.hidden_layers_text = QComboBox()
        self.hidden_layers_text.addItems(["64,32", "128,64", "256,128,64", "128,128,64,32"])
        self.hidden_layers_text.setCurrentIndex(3)  # 默认选择复杂网络
        hidden_layout.addWidget(self.hidden_layers_text)
        model_select_layout.addLayout(hidden_layout)

        # 初始学习率
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("初始学习率:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.0005)
        lr_layout.addWidget(self.lr_spin)
        model_select_layout.addLayout(lr_layout)

        # 最大学习率
        max_lr_layout = QHBoxLayout()
        max_lr_layout.addWidget(QLabel("最大学习率:"))
        self.model_select_max_lr_spin = QDoubleSpinBox()
        self.model_select_max_lr_spin.setRange(0.0001, 0.5)
        self.model_select_max_lr_spin.setSingleStep(0.0001)
        self.model_select_max_lr_spin.setDecimals(4)
        self.model_select_max_lr_spin.setValue(0.01)
        max_lr_layout.addWidget(self.model_select_max_lr_spin)
        model_select_layout.addLayout(max_lr_layout)

        # 折扣率
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("折扣率:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.8, 0.999)
        self.gamma_spin.setSingleStep(0.001)
        self.gamma_spin.setDecimals(3)
        self.gamma_spin.setValue(0.99)
        gamma_layout.addWidget(self.gamma_spin)
        model_select_layout.addLayout(gamma_layout)

        # 批量大小
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("批量大小:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 512)
        self.batch_size_spin.setSingleStep(16)
        self.batch_size_spin.setValue(64)
        batch_layout.addWidget(self.batch_size_spin)
        model_select_layout.addLayout(batch_layout)

        # 评估频率
        eval_freq_layout = QHBoxLayout()
        eval_freq_layout.addWidget(QLabel("评估频率:"))
        self.eval_freq_spin = QSpinBox()
        self.eval_freq_spin.setRange(1, 100)
        self.eval_freq_spin.setSingleStep(5)
        self.eval_freq_spin.setValue(20)
        eval_freq_layout.addWidget(self.eval_freq_spin)
        model_select_layout.addLayout(eval_freq_layout)

        model_group.setLayout(model_select_layout)
        model_layout.addWidget(model_group)

        # ==================== 环境配置标签页 ====================
        self.env_config_tab = QWidget()
        env_layout = QVBoxLayout(self.env_config_tab)

        # 环境参数
        env_group = QGroupBox("环境参数")
        env_params_layout = QVBoxLayout()

        # 窗口大小
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("观察窗口大小:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 60)
        self.window_spin.setSingleStep(5)
        self.window_spin.setValue(20)
        window_layout.addWidget(self.window_spin)
        env_params_layout.addLayout(window_layout)

        # 初始资金
        balance_layout = QHBoxLayout()
        balance_layout.addWidget(QLabel("初始资金:"))
        self.balance_spin = QSpinBox()
        self.balance_spin.setRange(1000, 100000)
        self.balance_spin.setSingleStep(1000)
        self.balance_spin.setValue(10000)
        balance_layout.addWidget(self.balance_spin)
        env_params_layout.addLayout(balance_layout)

        # 交易手续费
        fee_layout = QHBoxLayout()
        fee_layout.addWidget(QLabel("交易手续费(%):"))
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0, 1)
        self.fee_spin.setSingleStep(0.01)
        self.fee_spin.setDecimals(3)
        self.fee_spin.setValue(0.05)  # 0.05%
        fee_layout.addWidget(self.fee_spin)
        env_params_layout.addLayout(fee_layout)

        # 最大仓位比例
        max_position_layout = QHBoxLayout()
        max_position_layout.addWidget(QLabel("最大仓位比例(%):"))
        self.max_position_spin = QDoubleSpinBox()
        self.max_position_spin.setRange(1, 100)
        self.max_position_spin.setSingleStep(1)
        self.max_position_spin.setValue(100)
        max_position_layout.addWidget(self.max_position_spin)
        env_params_layout.addLayout(max_position_layout)

        # 基础仓位比例
        base_position_layout = QHBoxLayout()
        base_position_layout.addWidget(QLabel("基础仓位比例(%):"))
        self.base_position_spin = QDoubleSpinBox()
        self.base_position_spin.setRange(1, 100)
        self.base_position_spin.setSingleStep(1)
        self.base_position_spin.setValue(20)
        base_position_layout.addWidget(self.base_position_spin)
        env_params_layout.addLayout(base_position_layout)

        # 最大交易金额比例
        max_trade_amount_layout = QHBoxLayout()
        max_trade_amount_layout.addWidget(QLabel("最大交易金额比例(%):"))
        self.max_trade_amount_spin = QDoubleSpinBox()
        self.max_trade_amount_spin.setRange(1, 100)
        self.max_trade_amount_spin.setSingleStep(1)
        self.max_trade_amount_spin.setValue(50)
        max_trade_amount_layout.addWidget(self.max_trade_amount_spin)
        env_params_layout.addLayout(max_trade_amount_layout)

        # 最大回合步数
        max_steps_layout = QHBoxLayout()
        max_steps_layout.addWidget(QLabel("最大回合步数:"))
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(100, 10000)
        self.max_steps_spin.setValue(5000)  # 修改默认值为5000
        self.max_steps_spin.setSingleStep(100)
        max_steps_layout.addWidget(self.max_steps_spin)
        env_params_layout.addLayout(max_steps_layout)

        # 提前停止亏损阈值
        stop_loss_layout = QHBoxLayout()
        stop_loss_layout.addWidget(QLabel("提前停止亏损阈值(%):"))
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(5, 50)
        self.stop_loss_spin.setSingleStep(1)
        self.stop_loss_spin.setDecimals(1)
        self.stop_loss_spin.setValue(15)  # 15%
        stop_loss_layout.addWidget(self.stop_loss_spin)
        env_params_layout.addLayout(stop_loss_layout)

        # 启用提前停止
        self.early_stop_check = QCheckBox("启用提前停止")
        self.early_stop_check.setChecked(True)
        env_params_layout.addWidget(self.early_stop_check)

        # 技术指标
        self.use_indicators_check = QCheckBox("使用技术指标 (MA, RSI, MACD, BB)")
        self.use_indicators_check.setChecked(True)
        env_params_layout.addWidget(self.use_indicators_check)

        # 持仓信息
        self.include_position_check = QCheckBox("在状态中包含持仓信息")
        self.include_position_check.setChecked(True)
        env_params_layout.addWidget(self.include_position_check)

        # 惩罚不行动
        self.penalize_inaction_check = QCheckBox("惩罚长时间不交易")
        self.penalize_inaction_check.setChecked(True)
        env_params_layout.addWidget(self.penalize_inaction_check)

        # 启用仓位管理
        self.position_sizing_check = QCheckBox("启用仓位管理")
        self.position_sizing_check.setChecked(True)
        env_params_layout.addWidget(self.position_sizing_check)

        # 添加固定交易金额设置
        fixed_amount_layout = QHBoxLayout()
        fixed_amount_layout.addWidget(QLabel("固定交易金额:"))
        self.fixed_amount_spin = QDoubleSpinBox()
        self.fixed_amount_spin.setRange(1, 1000)
        self.fixed_amount_spin.setSingleStep(1)
        self.fixed_amount_spin.setValue(10.0)  # 默认10.0
        fixed_amount_layout.addWidget(self.fixed_amount_spin)
        env_params_layout.addLayout(fixed_amount_layout)

        # 最小交易间隔
        min_interval_layout = QHBoxLayout()
        min_interval_layout.addWidget(QLabel("最小交易间隔(步数):"))
        self.min_interval_spin = QSpinBox()
        self.min_interval_spin.setRange(1, 20)
        self.min_interval_spin.setSingleStep(1)
        self.min_interval_spin.setValue(5)  # 默认5步
        min_interval_layout.addWidget(self.min_interval_spin)
        env_params_layout.addLayout(min_interval_layout)

        # 每回合最大交易次数
        max_trades_layout = QHBoxLayout()
        max_trades_layout.addWidget(QLabel("每回合最大交易次数:"))
        self.max_trades_spin = QSpinBox()
        self.max_trades_spin.setRange(5, 100)
        self.max_trades_spin.setSingleStep(5)
        self.max_trades_spin.setValue(20)  # 默认20次
        max_trades_layout.addWidget(self.max_trades_spin)
        env_params_layout.addLayout(max_trades_layout)

        env_group.setLayout(env_params_layout)
        env_layout.addWidget(env_group)

        # ==================== 奖励设计标签页 ====================
        self.reward_design_tab = QWidget()
        reward_layout = QVBoxLayout(self.reward_design_tab)

        # 奖励设计
        reward_group = QGroupBox("奖励函数设计")
        reward_design_layout = QVBoxLayout()

        # 奖励类型
        reward_type_layout = QHBoxLayout()
        reward_type_layout.addWidget(QLabel("奖励类型:"))
        self.reward_type_combo = QComboBox()
        self.reward_type_combo.addItems(["利润", "夏普比率", "复合奖励"])
        self.reward_type_combo.setCurrentIndex(2)  # 默认使用复合奖励
        self.reward_type_combo.currentIndexChanged.connect(self.on_reward_type_changed)
        reward_type_layout.addWidget(self.reward_type_combo)

        # 添加奖励类型帮助按钮
        reward_type_help_btn = QPushButton("?")
        reward_type_help_btn.setFixedSize(20, 20)
        reward_type_help_btn.clicked.connect(lambda: QMessageBox.information(self, "奖励类型说明",
            "可选择的奖励计算方式：\n\n" \
            "1. 利润：直接使用交易盈亏作为奖励\n" \
            "   - 优点：直观反映交易效果\n" \
            "   - 缺点：可能导致过度追求短期利润\n\n" \
            "2. 夏普比率：考虑收益和风险的平衡\n" \
            "   - 优点：平衡收益和风险\n" \
            "   - 缺点：计算相对复杂\n\n" \
            "3. 复合奖励：综合多个因素\n" \
            "   - 包含：利润、累积收益、风险控制、趋势跟随等\n" \
            "   - 优点：全面评估交易表现\n" \
            "   - 缺点：需要仔细调整各组成部分的权重"))
        reward_type_layout.addWidget(reward_type_help_btn)
        reward_type_layout.addStretch()
        reward_design_layout.addLayout(reward_type_layout)

        # 复合奖励设置
        self.compound_reward_group = QGroupBox("复合奖励权重")
        compound_reward_layout = QVBoxLayout()

        # 利润奖励
        profit_layout = QHBoxLayout()
        profit_layout.addWidget(QLabel("利润奖励:"))
        self.profit_weight_spin = QDoubleSpinBox()
        self.profit_weight_spin.setRange(0, 5)
        self.profit_weight_spin.setSingleStep(0.1)
        self.profit_weight_spin.setDecimals(1)
        self.profit_weight_spin.setValue(1.0)
        profit_layout.addWidget(self.profit_weight_spin)

        # 添加利润奖励帮助按钮
        profit_help_btn = QPushButton("?")
        profit_help_btn.setFixedSize(20, 20)
        profit_help_btn.clicked.connect(lambda: QMessageBox.information(self, "利润奖励说明",
            "利润奖励是最基础的奖励组成部分：\n\n" \
            "计算方式：\n" \
            "1. 基于每笔交易的盈亏计算\n" \
            "2. 收益率 = (当前资产 - 上一步资产) / 上一步资产\n" \
            "3. 放大10倍使奖励信号更明显\n" \
            "4. 受最大单步奖励限制约束\n\n" \
            "权重说明：\n" \
            "- 权重越大，模型越注重短期盈利\n" \
            "- 建议范围：0.5-2.0"))
        profit_layout.addWidget(profit_help_btn)
        profit_layout.addStretch()
        compound_reward_layout.addLayout(profit_layout)

        # 累积收益奖励
        cum_return_layout = QHBoxLayout()
        cum_return_layout.addWidget(QLabel("累积收益奖励:"))
        self.cum_return_weight_spin = QDoubleSpinBox()
        self.cum_return_weight_spin.setRange(0, 5)
        self.cum_return_weight_spin.setSingleStep(0.1)
        self.cum_return_weight_spin.setDecimals(1)
        self.cum_return_weight_spin.setValue(2.0)
        cum_return_layout.addWidget(self.cum_return_weight_spin)

        # 添加累积收益奖励帮助按钮
        cum_return_help_btn = QPushButton("?")
        cum_return_help_btn.setFixedSize(20, 20)
        cum_return_help_btn.clicked.connect(lambda: QMessageBox.information(self, "累积收益奖励说明",
            "累积收益奖励关注长期表现：\n\n" \
            "计算方式：\n" \
            "1. 基于整个回合的总收益率\n" \
            "2. 考虑所有交易的累积效果\n" \
            "3. 使用对数收益率避免极端值\n\n" \
            "权重说明：\n" \
            "- 权重越大，模型越注重长期稳定收益\n" \
            "- 建议范围：1.0-3.0\n" \
            "- 通常设置大于利润奖励权重"))
        cum_return_layout.addWidget(cum_return_help_btn)
        cum_return_layout.addStretch()
        compound_reward_layout.addLayout(cum_return_layout)

        # 风险调整奖励
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(QLabel("风险调整奖励:"))
        self.risk_weight_spin = QDoubleSpinBox()
        self.risk_weight_spin.setRange(0, 5)
        self.risk_weight_spin.setSingleStep(0.1)
        self.risk_weight_spin.setDecimals(1)
        self.risk_weight_spin.setValue(1.5)
        risk_layout.addWidget(self.risk_weight_spin)

        # 添加风险调整奖励帮助按钮
        risk_help_btn = QPushButton("?")
        risk_help_btn.setFixedSize(20, 20)
        risk_help_btn.clicked.connect(lambda: QMessageBox.information(self, "风险调整奖励说明",
            "风险调整奖励用于控制交易风险：\n\n" \
            "计算方式：\n" \
            "1. 考虑收益的波动性\n" \
            "2. 计算夏普比率类型的指标\n" \
            "3. 对大幅波动进行惩罚\n\n" \
            "权重说明：\n" \
            "- 权重越大，模型越倾向于稳健交易\n" \
            "- 建议范围：1.0-2.0\n" \
            "- 需要与回撤惩罚配合使用"))
        risk_layout.addWidget(risk_help_btn)
        risk_layout.addStretch()
        compound_reward_layout.addLayout(risk_layout)

        # 趋势跟随奖励
        trend_layout = QHBoxLayout()
        trend_layout.addWidget(QLabel("趋势跟随奖励:"))
        self.trend_weight_spin = QDoubleSpinBox()
        self.trend_weight_spin.setRange(0, 5)
        self.trend_weight_spin.setSingleStep(0.1)
        self.trend_weight_spin.setDecimals(1)
        self.trend_weight_spin.setValue(0.8)
        trend_layout.addWidget(self.trend_weight_spin)

        # 添加趋势跟随奖励帮助按钮
        trend_help_btn = QPushButton("?")
        trend_help_btn.setFixedSize(20, 20)
        trend_help_btn.clicked.connect(lambda: QMessageBox.information(self, "趋势跟随奖励说明",
            "趋势跟随奖励鼓励顺势而为：\n\n" \
            "计算方式：\n" \
            "1. 判断价格趋势方向\n" \
            "2. 当交易方向与趋势一致时给予奖励\n" \
            "3. 考虑趋势的强度\n\n" \
            "权重说明：\n" \
            "- 权重越大，模型越倾向于追随趋势\n" \
            "- 建议范围：0.5-1.5\n" \
            "- 过高可能导致追涨杀跌"))
        trend_layout.addWidget(trend_help_btn)
        trend_layout.addStretch()
        compound_reward_layout.addLayout(trend_layout)

        # 最大单步奖励限制
        max_reward_layout = QHBoxLayout()
        max_reward_layout.addWidget(QLabel("最大单步奖励限制:"))
        self.max_reward_spin = QDoubleSpinBox()
        self.max_reward_spin.setRange(0.1, 2.0)
        self.max_reward_spin.setSingleStep(0.1)
        self.max_reward_spin.setDecimals(2)
        self.max_reward_spin.setValue(0.5)
        max_reward_layout.addWidget(self.max_reward_spin)

        # 添加最大单步奖励限制帮助按钮
        max_reward_help_btn = QPushButton("?")
        max_reward_help_btn.setFixedSize(20, 20)
        max_reward_help_btn.clicked.connect(lambda: QMessageBox.information(self, "最大单步奖励限制说明",
            "限制单次交易的最大奖励幅度：\n\n" \
            "作用：\n" \
            "1. 防止极端收益导致的过度学习\n" \
            "2. 平滑奖励信号\n" \
            "3. 控制模型行为的剧烈程度\n\n" \
            "具体实现：\n" \
            "- 对计算得到的奖励值进行截断\n" \
            "- 正向奖励上限：+0.5\n" \
            "- 负向惩罚下限：-0.5\n" \
            "- 建议范围：0.2-1.0"))
        max_reward_layout.addWidget(max_reward_help_btn)
        max_reward_layout.addStretch()
        compound_reward_layout.addLayout(max_reward_layout)

        # 成功交易基础奖励
        profit_base_reward_layout = QHBoxLayout()
        profit_base_reward_layout.addWidget(QLabel("成功交易基础奖励:"))
        self.profit_base_reward_spin = QDoubleSpinBox()
        self.profit_base_reward_spin.setRange(0.0, 0.5)
        self.profit_base_reward_spin.setSingleStep(0.01)
        self.profit_base_reward_spin.setDecimals(2)
        self.profit_base_reward_spin.setValue(0.05)
        profit_base_reward_layout.addWidget(self.profit_base_reward_spin)

        # 添加成功交易基础奖励帮助按钮
        profit_base_reward_help_btn = QPushButton("?")
        profit_base_reward_help_btn.setFixedSize(20, 20)
        profit_base_reward_help_btn.clicked.connect(lambda: QMessageBox.information(self, "成功交易基础奖励说明",
            "为任何盈利交易提供固定奖励：\n\n" \
            "作用：\n" \
            "1. 确保即使是小额盈利也有明确正向反馈\n" \
            "2. 增强对成功交易方向的学习\n" \
            "3. 避免微小收益被忽略\n\n" \
            "具体实现：\n" \
            "- 任何净收益为正的交易都会获得此固定奖励\n" \
            "- 与收益率无关，只要盈利就有\n" \
            "- 建议范围：0.02-0.1\n" \
            "- 过高可能导致模型忽略收益率大小"))
        profit_base_reward_layout.addWidget(profit_base_reward_help_btn)
        profit_base_reward_layout.addStretch()
        compound_reward_layout.addLayout(profit_base_reward_layout)

        # 奖励放大因子
        reward_amplifier_layout = QHBoxLayout()
        reward_amplifier_layout.addWidget(QLabel("奖励放大因子:"))
        self.reward_amplifier_spin = QDoubleSpinBox()
        self.reward_amplifier_spin.setRange(1.0, 50.0)
        self.reward_amplifier_spin.setSingleStep(1.0)
        self.reward_amplifier_spin.setDecimals(1)
        self.reward_amplifier_spin.setValue(20.0)
        reward_amplifier_layout.addWidget(self.reward_amplifier_spin)

        # 添加奖励放大因子帮助按钮
        reward_amplifier_help_btn = QPushButton("?")
        reward_amplifier_help_btn.setFixedSize(20, 20)
        reward_amplifier_help_btn.clicked.connect(lambda: QMessageBox.information(self, "奖励放大因子说明",
            "放大收益率产生的奖励信号：\n\n" \
            "作用：\n" \
            "1. 增强小收益率产生的奖励信号\n" \
            "2. 使模型能够区分微小收益率差异\n" \
            "3. 避免收益率计算结果过小被忽略\n\n" \
            "具体实现：\n" \
            "- 收益率直接乘以此因子值\n" \
            "- 例如：0.1%收益率 × 20 = 2%奖励\n" \
            "- 建议范围：10-30\n" \
            "- 与成功交易基础奖励配合使用效果更佳"))
        reward_amplifier_layout.addWidget(reward_amplifier_help_btn)
        reward_amplifier_layout.addStretch()
        compound_reward_layout.addLayout(reward_amplifier_layout)

        # 趋势跟随奖励
        trend_follow_layout = QHBoxLayout()
        trend_follow_layout.addWidget(QLabel("趋势跟随奖励:"))
        self.trend_follow_spin = QDoubleSpinBox()
        self.trend_follow_spin.setRange(0.0, 1.0)
        self.trend_follow_spin.setSingleStep(0.05)
        self.trend_follow_spin.setDecimals(2)
        self.trend_follow_spin.setValue(0.3)
        trend_follow_layout.addWidget(self.trend_follow_spin)

        # 添加趋势跟随奖励帮助按钮
        trend_follow_help_btn = QPushButton("?")
        trend_follow_help_btn.setFixedSize(20, 20)
        trend_follow_help_btn.clicked.connect(lambda: QMessageBox.information(self, "趋势跟随奖励说明",
            "根据市场趋势调整奖励：\n\n" \
            "计算方式：\n" \
            "1. 分析价格走势和成交量\n" \
            "2. 识别市场趋势的强度\n" \
            "3. 当交易方向与趋势一致时增加奖励\n\n" \
            "参数说明：\n" \
            "- 值越大，趋势跟随效应越强\n" \
            "- 建议范围：0.1-0.5\n" \
            "- 过高可能错过趋势转折点"))
        trend_follow_layout.addWidget(trend_follow_help_btn)
        trend_follow_layout.addStretch()
        compound_reward_layout.addLayout(trend_follow_layout)

        self.compound_reward_group.setLayout(compound_reward_layout)
        reward_design_layout.addWidget(self.compound_reward_group)

        # 复合惩罚设置
        self.compound_penalty_group = QGroupBox("复合惩罚权重")
        compound_penalty_layout = QVBoxLayout()

        # 回撤惩罚
        drawdown_layout = QHBoxLayout()
        drawdown_layout.addWidget(QLabel("回撤惩罚:"))
        self.drawdown_weight_spin = QDoubleSpinBox()
        self.drawdown_weight_spin.setRange(-5, 0)
        self.drawdown_weight_spin.setSingleStep(0.1)
        self.drawdown_weight_spin.setDecimals(1)
        self.drawdown_weight_spin.setValue(-1.0)
        drawdown_layout.addWidget(self.drawdown_weight_spin)

        # 添加回撤惩罚帮助按钮
        drawdown_help_btn = QPushButton("?")
        drawdown_help_btn.setFixedSize(20, 20)
        drawdown_help_btn.clicked.connect(lambda: QMessageBox.information(self, "回撤惩罚说明",
            "对资金回撤进行惩罚：\n\n" \
            "计算方式：\n" \
            "1. 监控资金净值的下跌幅度\n" \
            "2. 根据回撤幅度给予负向奖励\n" \
            "3. 回撤越大，惩罚越重\n\n" \
            "权重说明：\n" \
            "- 范围：-5.0到0\n" \
            "- 建议值：-1.0到-2.0\n" \
            "- 过大可能导致过度保守"))
        drawdown_layout.addWidget(drawdown_help_btn)
        drawdown_layout.addStretch()
        compound_penalty_layout.addLayout(drawdown_layout)

        # 最大回撤惩罚
        max_drawdown_layout = QHBoxLayout()
        max_drawdown_layout.addWidget(QLabel("最大回撤惩罚:"))
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(-1.0, 0.0)
        self.max_drawdown_spin.setSingleStep(0.05)
        self.max_drawdown_spin.setDecimals(2)
        self.max_drawdown_spin.setValue(-0.3)
        max_drawdown_layout.addWidget(self.max_drawdown_spin)

        # 添加最大回撤惩罚帮助按钮
        max_drawdown_help_btn = QPushButton("?")
        max_drawdown_help_btn.setFixedSize(20, 20)
        max_drawdown_help_btn.clicked.connect(lambda: QMessageBox.information(self, "最大回撤惩罚说明",
            "针对最大历史回撤的特殊惩罚：\n\n" \
            "计算方式：\n" \
            "1. 记录训练过程中的最大回撤\n" \
            "2. 当超过阈值时给予额外惩罚\n" \
            "3. 用于控制极端风险\n\n" \
            "参数说明：\n" \
            "- 范围：-1.0到0\n" \
            "- 建议值：-0.3到-0.5\n" \
            "- 与普通回撤惩罚配合使用"))
        max_drawdown_layout.addWidget(max_drawdown_help_btn)
        max_drawdown_layout.addStretch()
        compound_penalty_layout.addLayout(max_drawdown_layout)

        # 交易频率控制
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("交易频率控制:"))
        self.freq_weight_spin = QDoubleSpinBox()
        self.freq_weight_spin.setRange(-5, 0)
        self.freq_weight_spin.setSingleStep(0.1)
        self.freq_weight_spin.setDecimals(1)
        self.freq_weight_spin.setValue(-0.5)
        freq_layout.addWidget(self.freq_weight_spin)
        compound_penalty_layout.addLayout(freq_layout)

        # 频繁交易惩罚
        frequent_trade_layout = QHBoxLayout()
        frequent_trade_layout.addWidget(QLabel("频繁交易惩罚:"))
        self.frequent_trade_spin = QDoubleSpinBox()
        self.frequent_trade_spin.setRange(-0.2, 0.0)
        self.frequent_trade_spin.setSingleStep(0.01)
        self.frequent_trade_spin.setDecimals(2)
        self.frequent_trade_spin.setValue(-0.05)
        frequent_trade_layout.addWidget(self.frequent_trade_spin)
        compound_penalty_layout.addLayout(frequent_trade_layout)

        # 不交易惩罚
        inaction_layout = QHBoxLayout()
        inaction_layout.addWidget(QLabel("不交易惩罚:"))
        self.inaction_spin = QDoubleSpinBox()
        self.inaction_spin.setRange(-0.2, 0.0)
        self.inaction_spin.setSingleStep(0.01)
        self.inaction_spin.setDecimals(2)
        self.inaction_spin.setValue(-0.05)
        inaction_layout.addWidget(self.inaction_spin)
        compound_penalty_layout.addLayout(inaction_layout)

        # 连续买入惩罚
        consecutive_buy_layout = QHBoxLayout()
        consecutive_buy_layout.addWidget(QLabel("连续买入惩罚:"))
        self.consecutive_buy_spin = QDoubleSpinBox()
        self.consecutive_buy_spin.setRange(-0.3, 0.0)
        self.consecutive_buy_spin.setSingleStep(0.01)
        self.consecutive_buy_spin.setDecimals(2)
        self.consecutive_buy_spin.setValue(-0.1)
        consecutive_buy_layout.addWidget(self.consecutive_buy_spin)
        compound_penalty_layout.addLayout(consecutive_buy_layout)

        # 趋势不一致惩罚
        trend_misalign_layout = QHBoxLayout()
        trend_misalign_layout.addWidget(QLabel("趋势不一致惩罚:"))
        self.trend_misalign_spin = QDoubleSpinBox()
        self.trend_misalign_spin.setRange(-0.3, 0.0)
        self.trend_misalign_spin.setSingleStep(0.01)
        self.trend_misalign_spin.setDecimals(2)
        self.trend_misalign_spin.setValue(-0.05)
        trend_misalign_layout.addWidget(self.trend_misalign_spin)
        compound_penalty_layout.addLayout(trend_misalign_layout)

        # 长时间持仓惩罚
        position_holding_layout = QHBoxLayout()
        position_holding_layout.addWidget(QLabel("长时间持仓惩罚:"))
        self.position_holding_spin = QDoubleSpinBox()
        self.position_holding_spin.setRange(-0.2, 0.0)
        self.position_holding_spin.setSingleStep(0.01)
        self.position_holding_spin.setDecimals(2)
        self.position_holding_spin.setValue(-0.05)
        position_holding_layout.addWidget(self.position_holding_spin)
        compound_penalty_layout.addLayout(position_holding_layout)

        self.compound_penalty_group.setLayout(compound_penalty_layout)
        reward_design_layout.addWidget(self.compound_penalty_group)

        # 交易间隔阈值
        trade_interval_layout = QHBoxLayout()
        trade_interval_layout.addWidget(QLabel("交易间隔阈值(步数):"))
        self.trade_interval_spin = QSpinBox()
        self.trade_interval_spin.setRange(1, 30)
        self.trade_interval_spin.setSingleStep(1)
        self.trade_interval_spin.setValue(10)
        trade_interval_layout.addWidget(self.trade_interval_spin)
        compound_penalty_layout.addLayout(trade_interval_layout)

        reward_group.setLayout(reward_design_layout)
        reward_layout.addWidget(reward_group)

        # ==================== 训练控制标签页 ====================
        self.training_config_tab = QWidget()
        training_config_layout = QVBoxLayout(self.training_config_tab)

        # 训练控制
        train_group = QGroupBox("训练控制")
        train_layout = QVBoxLayout()

        # 训练轮数
        episodes_layout = QHBoxLayout()
        episodes_layout.addWidget(QLabel("最大训练轮数:"))
        self.max_episodes_spin = QSpinBox()
        self.max_episodes_spin.setRange(10, 10000)
        self.max_episodes_spin.setSingleStep(10)
        self.max_episodes_spin.setValue(500)
        episodes_layout.addWidget(self.max_episodes_spin)
        train_layout.addLayout(episodes_layout)

        # 训练/验证数据比例
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("训练数据比例(%):"))
        self.train_ratio_spin = QSpinBox()
        self.train_ratio_spin.setRange(50, 90)
        self.train_ratio_spin.setSingleStep(5)
        self.train_ratio_spin.setValue(70)
        ratio_layout.addWidget(self.train_ratio_spin)
        train_layout.addLayout(ratio_layout)

        # 训练按钮
        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        buttons_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        self.eval_best_btn = QPushButton("评估最佳模型")
        self.eval_best_btn.clicked.connect(self.evaluate_best_model)
        self.eval_best_btn.setEnabled(False)
        buttons_layout.addWidget(self.eval_best_btn)

        train_layout.addLayout(buttons_layout)

        # 添加最佳模型信息
        self.best_model_info = QLabel("最佳模型: 无")
        train_layout.addWidget(self.best_model_info)

        train_group.setLayout(train_layout)
        training_config_layout.addWidget(train_group)

        # 训练状态
        status_group = QGroupBox("训练状态")
        status_layout = QVBoxLayout()

        # 状态标签
        self.status_label = QLabel("未开始训练")
        status_layout.addWidget(self.status_label)

        # 进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("训练进度:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        status_layout.addLayout(progress_layout)

        # 添加当前回合的进度条
        current_episode_layout = QHBoxLayout()
        current_episode_layout.addWidget(QLabel("当前回合进度:"))
        self.episode_progress_bar = QProgressBar()
        self.episode_progress_bar.setRange(0, 100)
        self.episode_progress_bar.setValue(0)
        current_episode_layout.addWidget(self.episode_progress_bar)
        status_layout.addLayout(current_episode_layout)

        self.episode_label = QLabel("回合: 0/0")
        status_layout.addWidget(self.episode_label)

        self.step_label = QLabel("当前步骤: 0/0")
        status_layout.addWidget(self.step_label)

        self.reward_label = QLabel("当前奖励: 0.0")
        status_layout.addWidget(self.reward_label)

        self.return_label = QLabel("收益率: 0.0%")
        status_layout.addWidget(self.return_label)

        # 添加学习率标签
        self.learning_rate_label = QLabel("学习率: 0.000500")
        status_layout.addWidget(self.learning_rate_label)

        status_group.setLayout(status_layout)
        training_config_layout.addWidget(status_group)

        # 添加标签页到配置标签页控件
        self.config_tabs.addTab(self.model_config_tab, "模型选择")
        self.config_tabs.addTab(self.env_config_tab, "环境配置")
        self.config_tabs.addTab(self.reward_design_tab, "奖励设计")
        self.config_tabs.addTab(self.training_config_tab, "训练控制")

        # 将标签页控件添加到左侧面板
        config_layout.addWidget(self.config_tabs)

        # 添加左侧面板到分割器
        self.splitter.addWidget(self.config_panel)

        # ==================== 右侧结果面板 ====================
        self.results_panel = QWidget()
        results_layout = QVBoxLayout(self.results_panel)

        # 初始化结果标签页
        results_tabs = self.init_result_tabs()
        results_layout.addWidget(results_tabs)

        # 添加日志区域
        self.log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.log_group.setLayout(log_layout)
        results_layout.addWidget(self.log_group)

        # 添加右侧面板到分割器
        self.splitter.addWidget(self.results_panel)

        # 设置分割器初始比例
        self.splitter.setSizes([300, 700])  # 左侧:右侧 = 3:7

        # 将分割器添加到主布局
        main_layout.addWidget(self.splitter)

        # 尝试初始化matplotlib图表
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            self.matplotlib_available = True
            self.init_plots()
        except ImportError:
            self.matplotlib_available = False
            self.log_message("警告: matplotlib不可用，无法显示图表")

    def init_result_tabs(self):
        """初始化结果标签页"""
        # 添加标签页
        self.results_tabs = QTabWidget()

        # 添加图表控制组
        self.chart_control_tab = QWidget()
        chart_control_layout = QVBoxLayout(self.chart_control_tab)

        # 创建图表控制组框
        chart_control_group = QGroupBox("图表显示控制")
        chart_control_inner_layout = QVBoxLayout()

        # 添加复选框
        self.show_rewards_check = QCheckBox("显示奖励曲线")
        self.show_rewards_check.setChecked(True)
        self.show_rewards_check.stateChanged.connect(self.on_chart_visibility_changed)
        chart_control_inner_layout.addWidget(self.show_rewards_check)

        self.show_returns_check = QCheckBox("显示收益曲线")
        self.show_returns_check.setChecked(True)
        self.show_returns_check.stateChanged.connect(self.on_chart_visibility_changed)
        chart_control_inner_layout.addWidget(self.show_returns_check)

        self.show_learning_rate_check = QCheckBox("显示学习率曲线")
        self.show_learning_rate_check.setChecked(True)
        self.show_learning_rate_check.stateChanged.connect(self.on_chart_visibility_changed)
        chart_control_inner_layout.addWidget(self.show_learning_rate_check)

        self.show_portfolio_check = QCheckBox("显示资产曲线")
        self.show_portfolio_check.setChecked(True)
        self.show_portfolio_check.stateChanged.connect(self.on_chart_visibility_changed)
        chart_control_inner_layout.addWidget(self.show_portfolio_check)

        chart_control_group.setLayout(chart_control_inner_layout)
        chart_control_layout.addWidget(chart_control_group)

        # 添加说明文本
        chart_info_label = QLabel("注意：取消选中图表将停止其更新，可以减少训练过程中的计算负担。")
        chart_info_label.setWordWrap(True)
        chart_control_layout.addWidget(chart_info_label)

        # 添加弹性空间
        chart_control_layout.addStretch()

        # ==================== 奖励曲线标签页 ====================
        self.rewards_tab = QWidget()
        rewards_layout = QVBoxLayout(self.rewards_tab)

        self.rewards_plot_widget = QWidget()
        rewards_layout.addWidget(self.rewards_plot_widget)

        # ==================== 收益曲线标签页 ====================
        self.returns_tab = QWidget()
        returns_layout = QVBoxLayout(self.returns_tab)

        self.returns_plot_widget = QWidget()
        returns_layout.addWidget(self.returns_plot_widget)

        # ==================== 最佳模型评估标签页 ====================
        self.best_model_tab = QWidget()
        best_model_layout = QVBoxLayout(self.best_model_tab)

        # 添加最佳模型收益曲线图
        self.best_model_plot_widget = QWidget()
        self.best_model_plot_layout = QVBoxLayout(self.best_model_plot_widget)
        self.best_model_plot_label = QLabel("评估后将显示最佳模型的收益曲线...")
        self.best_model_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.best_model_plot_layout.addWidget(self.best_model_plot_label)

        best_model_layout.addWidget(self.best_model_plot_widget)

        # 添加性能指标表格
        self.best_model_metrics_group = QGroupBox("最佳模型性能指标")
        best_metrics_layout = QVBoxLayout()

        self.best_metrics_text = QTextEdit()
        self.best_metrics_text.setReadOnly(True)
        best_metrics_layout.addWidget(self.best_metrics_text)

        self.best_model_metrics_group.setLayout(best_metrics_layout)
        best_model_layout.addWidget(self.best_model_metrics_group)

        # ==================== 性能指标标签页 ====================
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)

        # 性能指标面板
        self.metrics_group = QGroupBox("奖惩日志")
        metrics_chart_layout = QVBoxLayout()

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_chart_layout.addWidget(self.metrics_text)

        self.metrics_group.setLayout(metrics_chart_layout)
        metrics_layout.addWidget(self.metrics_group)

        # ==================== 训练交易记录标签页 ====================
        self.training_trades_tab = QWidget()
        training_trades_layout = QVBoxLayout(self.training_trades_tab)

        # 添加交易记录表格
        self.training_trades_table = QTableWidget()
        self.training_trades_table.setColumnCount(10)
        self.training_trades_table.setHorizontalHeaderLabels([
            '时间', '操作', '价格', '数量', '交易金额', '手续费', '收益', '收益率(%)', '余额', '总资产'
        ])

        # 设置表格样式
        self.training_trades_table.setAlternatingRowColors(True)
        self.training_trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.training_trades_table.verticalHeader().setVisible(True)
        self.training_trades_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        training_trades_layout.addWidget(self.training_trades_table)

        # ==================== 评估交易记录标签页 ====================
        self.evaluation_trades_tab = QWidget()
        # 使用垂直布局，移除右侧日志区域
        evaluation_trades_layout = QVBoxLayout(self.evaluation_trades_tab)

        # 添加交易记录表格，使用全宽显示
        self.evaluation_trades_table = QTableWidget()
        self.evaluation_trades_table.setColumnCount(10)
        self.evaluation_trades_table.setHorizontalHeaderLabels([
            '时间', '操作', '价格', '数量', '交易金额', '手续费', '收益', '收益率(%)', '余额', '总资产'
        ])

        # 设置表格样式
        self.evaluation_trades_table.setAlternatingRowColors(True)
        self.evaluation_trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.evaluation_trades_table.verticalHeader().setVisible(True)
        self.evaluation_trades_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        evaluation_trades_layout.addWidget(self.evaluation_trades_table)

        # ==================== 训练日志标签页 ====================
        self.log_tab = QWidget()
        log_tab_layout = QVBoxLayout(self.log_tab)

        # 添加日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_tab_layout.addWidget(self.log_text)

        # 添加学习率曲线标签页
        # 创建学习率曲线标签页
        self.learning_rate_tab = QWidget()
        learning_rate_layout = QVBoxLayout(self.learning_rate_tab)

        # 创建学习率曲线图表
        from rl_strategies.ui.learning_rate_plot import LearningRatePlot
        self.learning_rate_plot = LearningRatePlot()
        self.learning_rate_plot.setMinimumSize(400, 300)  # 设置最小尺寸确保图表可见
        learning_rate_layout.addWidget(self.learning_rate_plot)

        # 创建水平布局来放置学习率和探索率信息
        info_layout = QHBoxLayout()
        
        # 添加学习率信息标签
        self.learning_rate_info = QLabel("当前学习率: 0.000500")
        info_layout.addWidget(self.learning_rate_info)
        
        # 添加探索率信息标签
        self.epsilon_info = QLabel("当前探索率: 1.000000")
        info_layout.addWidget(self.epsilon_info)
        
        # 将水平布局添加到主布局
        learning_rate_layout.addLayout(info_layout)

        # 添加探索率标签页
        self.epsilon_tab = QWidget()
        epsilon_layout = QVBoxLayout(self.epsilon_tab)

        # 添加探索率文本框
        self.epsilon_text = QTextEdit()
        self.epsilon_text.setReadOnly(True)
        self.epsilon_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)  # 禁用自动换行
        self.epsilon_text.setFont(QApplication.font("Monospace"))  # 使用等宽字体
        epsilon_layout.addWidget(self.epsilon_text)

        # 将所有标签页添加到结果标签页控件
        self.results_tabs.addTab(self.chart_control_tab, "图表控制")
        self.results_tabs.addTab(self.rewards_tab, "奖励曲线")
        self.results_tabs.addTab(self.returns_tab, "收益曲线")
        self.results_tabs.addTab(self.learning_rate_tab, "学习率曲线")
        self.results_tabs.addTab(self.epsilon_tab, "探索率")
        self.results_tabs.addTab(self.metrics_tab, "奖惩日志")
        self.results_tabs.addTab(self.best_model_tab, "最佳模型")
        self.results_tabs.addTab(self.training_trades_tab, "训练交易记录")
        self.results_tabs.addTab(self.evaluation_trades_tab, "评估交易记录")
        self.results_tabs.addTab(self.log_tab, "训练日志")

        return self.results_tabs

    def on_epsilon_change(self, epsilon_message):
        """
        处理探索率变化信息 - 已弃用，现在由update_epsilon_info直接处理
        保留此方法以兼容可能的旧代码调用
        
        参数:
            epsilon_message: 探索率变化信息字符串
        """
        # 此方法已被update_epsilon_info替代，不再需要实现
        # 为了兼容性，保留此方法但不执行任何操作
        pass
    
    def unified_data_callback(self, data, source_type='training'):
        """
        统一的数据回调处理函数，处理来自训练和评估的数据

        参数:
            data: 包含更新数据的字典
            source_type: 数据来源类型，'training'或'evaluation'
        """
        try:
            # 共同处理部分
            if 'episode' in data:
                episode = data['episode']
                print(f"UI收到{source_type}数据，回合：{episode}")

            # 训练数据特定处理
            if source_type == 'training':
                # 处理训练进度更新
                self.handle_training_progress(data)

                # 处理训练交易记录
                if 'trade_records' in data and data['trade_records']:
                    # 保存训练交易记录
                    self.training_trades = data['trade_records']
                    # 更新训练交易表格
                    self.update_training_trade_table(self.training_trades)

            # 评估数据特定处理
            elif source_type == 'evaluation':
                # 处理评估结果
                self.handle_evaluation_result(data)

                # 处理评估交易记录
                trade_data = None
                # 查找交易记录
                if 'trade_history' in data and data['trade_history']:
                    trade_data = data['trade_history']
                elif 'trades' in data and data['trades']:
                    trade_data = data['trades']

                if trade_data:
                    # 保存评估交易记录
                    self.evaluation_trades = trade_data
                    # 更新评估交易表格
                    self.update_evaluation_trade_table(self.evaluation_trades)

            # 强制更新UI
            QApplication.processEvents()

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"处理{source_type}数据时出错: {str(e)}\n{error_msg}")
            self.log_message(f"处理{source_type}数据时出错: {str(e)}")

    def handle_training_progress(self, data):
        """处理训练进度数据"""
        # 判断是否需要更新图表
        need_update_charts = False

        # 确保rewards列表存在，即使数据中没有
        if not hasattr(self, 'accumulated_rewards'):
            self.accumulated_rewards = []

        # 确保rewards_history存在，用于存储整个训练过程的奖励数据
        if not hasattr(self, 'rewards_history'):
            self.rewards_history = []

        # 确保learning_rates_history存在，用于存储学习率变化数据
        if not hasattr(self, 'learning_rates_history'):
            self.learning_rates_history = []
            self.learning_rate_steps = []

        # 更新总进度条
        if 'episode' in data and 'max_episodes' in data:
            episode = data['episode']
            max_episodes = data['max_episodes']
            progress = int(episode / max_episodes * 100) if max_episodes > 0 else 0
            self.progress_bar.setValue(progress)
            self.episode_label.setText(f"回合: {episode}/{max_episodes}")

        # 更新当前回合进度和步数计数器
        current_step = None
        if 'step' in data and 'max_steps' in data:
            step = data['step']
            current_step = step  # 记录当前步数
            max_steps = data['max_steps']
            
            # 更新UI侧的步数计数器
            self.ui_step_counter = step
            print(f"[训练进度] UI步数计数器更新为: {self.ui_step_counter}")
            
            if max_steps > 0:
                episode_progress = int(step / max_steps * 100)
                self.episode_progress_bar.setValue(episode_progress)
                self.step_label.setText(f"当前步骤: {step}/{max_steps}")
        
        # 如果有环境信息，也尝试更新步数
        if 'env_info' in data and isinstance(data['env_info'], dict):
            env_info = data['env_info']
            if 'current_step' in env_info:
                env_step = env_info['current_step']
                # 只有当环境步数大于当前UI步数时才更新
                if env_step > self.ui_step_counter:
                    self.ui_step_counter = env_step
                    print(f"[环境信息] UI步数计数器更新为: {self.ui_step_counter}")
                    
        # 如果有代理信息，尝试更新步数和探索率
        if 'agent_info' in data and isinstance(data['agent_info'], dict):
            agent_info = data['agent_info']
            
            # 更新步数
            if 'learn_step_counter' in agent_info:
                agent_step = agent_info['learn_step_counter']
                # 只有当代理步数大于当前UI步数时才更新
                if agent_step > self.ui_step_counter:
                    self.ui_step_counter = agent_step
                    print(f"[代理信息] UI步数计数器更新为: {self.ui_step_counter}")
            
            # 更新探索率
            if 'epsilon' in agent_info:
                agent_epsilon = agent_info['epsilon']
                # 只有当代理探索率大于0时才更新
                if agent_epsilon > 0.0:
                    self.ui_epsilon_value = agent_epsilon
                    print(f"[代理信息] UI探索率更新为: {self.ui_epsilon_value}")
                else:
                    print(f"[代理信息] 忽略代理探索率为0的更新: {agent_epsilon}")
        
        # 如果数据中直接包含epsilon信息
        if 'epsilon' in data:
            data_epsilon = data['epsilon']
            if data_epsilon > 0.0:
                self.ui_epsilon_value = data_epsilon
                print(f"[训练进度] UI探索率直接更新为: {self.ui_epsilon_value}")
            else:
                print(f"[训练进度] 忽略数据中探索率为0的更新: {data_epsilon}")
                
        # 处理学习率数据
        learning_rate_updated = False
        if 'learning_rate' in data:
            current_lr = data['learning_rate']
            self.learning_rate_label.setText(f"学习率: {current_lr:.6f}")

            # 如果是第一个数据点，或者学习率发生变化，则记录
            if (len(self.learning_rates_history) == 0 or
                current_lr != self.learning_rates_history[-1]):

                # 使用实际的步数
                if 'step' in data:
                    step_to_record = data['step']  # 直接使用当前步数
                else:
                    # 如果没有当前步数信息，使用最后记录的步数+10或默认为10
                    step_to_record = (self.learning_rate_steps[-1] + 10
                                    if self.learning_rate_steps else 10)

                self.learning_rates_history.append(current_lr)
                self.learning_rate_steps.append(step_to_record)

                # 检查是否需要进行数据压缩
                if self.enable_data_compression and len(self.learning_rates_history) > self.compress_threshold:
                    compressed_rates, compressed_steps = self.compress_data(
                        self.learning_rates_history, self.learning_rate_steps)
                    self.learning_rates_history = compressed_rates
                    self.learning_rate_steps = compressed_steps

                learning_rate_updated = True

        # 如果收到learning_rates列表，直接更新整个历史
        if 'learning_rates' in data and data['learning_rates']:
            new_learning_rates = data['learning_rates']

            # 如果提供了对应的步数
            if 'learning_rate_steps' in data and len(data['learning_rate_steps']) == len(new_learning_rates):
                new_steps = data['learning_rate_steps']
            else:
                # 如果有当前步数，使用它来生成合适的步数序列
                current_step = data.get('step', 0)
                step_interval = current_step / (len(new_learning_rates) - 1) if len(new_learning_rates) > 1 else current_step
                new_steps = [int(i * step_interval) for i in range(len(new_learning_rates))]

            # 只有在数据量增加或最后一个值变化时才更新
            if (len(new_learning_rates) > len(self.learning_rates_history) or
                (len(new_learning_rates) > 0 and len(self.learning_rates_history) > 0 and
                 new_learning_rates[-1] != self.learning_rates_history[-1])):

                self.learning_rates_history = new_learning_rates
                self.learning_rate_steps = new_steps
                learning_rate_updated = True

        # 更新奖励标签
        if 'reward' in data:
            reward = data['reward']
            self.reward_label.setText(f"当前奖励: {reward:.4f}")

            # 累积奖励数据（如果数据中没有完整的rewards列表）
            if reward != 0 and (len(self.accumulated_rewards) == 0 or reward != self.accumulated_rewards[-1]):
                self.accumulated_rewards.append(reward)

        if 'return' in data:
            ret = data['return']
            self.return_label.setText(f"收益率: {ret:.2f}%")

        # 如果有最佳模型更新
        if 'best_model' in data and data['best_model']:
            self.best_model = data['best_model']
            self.best_model_reward = data.get('best_reward', 0)
            self.best_model_episode = data.get('best_episode', 0)

            # 存储额外的最佳模型信息
            if 'best_composite_score' in data:
                self.best_composite_score = data['best_composite_score']
            if 'best_eval_return' in data:
                self.best_eval_return = data['best_eval_return']
            if 'best_win_rate' in data:
                self.best_win_rate = data['best_win_rate']

            # 更新UI显示
            model_info = f"最佳模型 (回合 {self.best_model_episode})"
            if hasattr(self, 'best_model_reward'):
                model_info += f" | 奖励: {self.best_model_reward:.4f}"
            if hasattr(self, 'best_composite_score'):
                model_info += f" | 综合得分: {self.best_composite_score:.4f}"

            self.best_model_info.setText(model_info)

            # 记录日志
            self.log_message(f"发现新的最佳模型: {model_info}")

        # ===== 处理rewards数据 =====
        rewards_updated = False
        if 'rewards' in data and data['rewards']:
            new_rewards = data['rewards']

            # 确保rewards_history存在
            if not hasattr(self, 'rewards_history'):
                self.rewards_history = []

            # 确保奖励曲线步数存在
            if not hasattr(self, 'rewards_steps'):
                self.rewards_steps = []

            # 如果有步数信息，生成步数序列
            # 奖励曲线数据是每10步采样一次
            rewards_steps = []
            if 'rewards_steps' in data and len(data['rewards_steps']) == len(new_rewards):
                # 使用提供的步数信息
                rewards_steps = data['rewards_steps']
            else:
                # 生成步数序列 - 每10步一个点
                for i in range(len(new_rewards)):
                    if i == 0:
                        rewards_steps.append(1)  # 第一个点是步数1
                    else:
                        rewards_steps.append(rewards_steps[i-1] + 10)  # 每10步一个点

            # 修改处理逻辑，确保不在新回合开始时清空历史数据
            if data.get('is_done', False):
                # 只有最终更新时才完全替换数据
                self.rewards_history = new_rewards.copy()
                self.rewards_steps = rewards_steps.copy()
                # 记录当前episode以便跟踪
                self.current_episode_rewards = data.get('episode', 0)
                rewards_updated = True

                # 输出最终累计奖励值
                if len(new_rewards) > 0:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] 累计奖励值: {new_rewards[-1]:.4f} (最终值)")
            else:
                # 检查数据更新情况
                needs_update = False

                # 情况1: 新数据比历史数据多
                if len(new_rewards) > len(self.rewards_history):
                    needs_update = True
                # 情况2: 数据量相同，但最后一个值有变化
                elif len(new_rewards) == len(self.rewards_history) and len(new_rewards) > 0:
                    if new_rewards[-1] != self.rewards_history[-1]:
                        needs_update = True
                # 情况3: 检测到新回合但有数据
                elif 'episode' in data and data['episode'] != getattr(self, 'current_episode_rewards', None) and len(new_rewards) > 0:
                    # 记录新回合但不重置数据
                    self.current_episode_rewards = data.get('episode', 0)
                    needs_update = True

                # 如果需要更新，则更新数据
                if needs_update:
                    self.rewards_history = new_rewards.copy()
                    self.rewards_steps = rewards_steps.copy()
                    rewards_updated = True

                    # 输出当前累计奖励值
                    if len(new_rewards) > 0:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        current_step = data.get('step', len(new_rewards) * 10)
                        print(f"[{current_time}] 累计奖励值: {new_rewards[-1]:.4f} (步数: {current_step})")
        elif self.accumulated_rewards and len(self.accumulated_rewards) > 1:
            # 如果没有收到完整rewards，使用累积的备用数据
            rewards_updated = True

            # 输出累积的备用奖励数据
            if len(self.accumulated_rewards) > 0:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] 累计奖励值(备用): {self.accumulated_rewards[-1]:.4f}")

        # ===== 处理returns数据 =====
        returns_updated = False
        if 'returns' in data:
            new_returns = data['returns']

            # 确保returns_history存在
            if not hasattr(self, 'returns_history'):
                self.returns_history = []

            # 确保收益曲线步数存在
            if not hasattr(self, 'returns_steps'):
                self.returns_steps = []

            # 如果有步数信息，生成步数序列
            # 收益曲线数据是每10步采样一次
            returns_steps = []
            if 'returns_steps' in data and len(data['returns_steps']) == len(new_returns):
                # 使用提供的步数信息
                returns_steps = data['returns_steps']
            else:
                # 生成步数序列 - 每10步一个点
                for i in range(len(new_returns)):
                    if i == 0:
                        returns_steps.append(1)  # 第一个点是步数1
                    else:
                        returns_steps.append(returns_steps[i-1] + 10)  # 每10步一个点

            # 修改处理逻辑，确保不在新回合开始时清空历史数据
            if data.get('is_done', False):
                # 只有最终更新时才完全替换数据
                self.returns_history = new_returns.copy()
                self.returns_steps = returns_steps.copy()
                # 记录当前episode以便跟踪
                self.current_episode_returns = data.get('episode', 0)
                returns_updated = True
            else:
                # 检查数据更新情况
                needs_update = False

                # 情况1: 新数据比历史数据多
                if len(new_returns) > len(self.returns_history):
                    needs_update = True
                # 情况2: 数据量相同，但最后一个值有变化
                elif len(new_returns) == len(self.returns_history) and len(new_returns) > 0:
                    if new_returns[-1] != self.returns_history[-1]:
                        needs_update = True
                # 情况3: 检测到新回合但有数据
                elif 'episode' in data and data['episode'] != getattr(self, 'current_episode_returns', None) and len(new_returns) > 0:
                    # 记录新回合但不重置数据
                    self.current_episode_returns = data.get('episode', 0)
                    needs_update = True

                # 如果需要更新，则更新数据
                if needs_update:
                    self.returns_history = new_returns.copy()
                    self.returns_steps = returns_steps.copy()
                    returns_updated = True

        # 在所有数据处理完成后，一次性更新所有图表
        if rewards_updated or returns_updated or learning_rate_updated:
            # 更新奖励曲线
            if rewards_updated and hasattr(self, 'rewards_history') and self.rewards_history:
                try:
                    self.update_rewards_plot(self.rewards_history)
                except Exception as e:
                    print(f"ERROR: 更新奖励曲线时出错: {str(e)}")
            elif rewards_updated and self.accumulated_rewards and len(self.accumulated_rewards) > 1:
                try:
                    self.update_rewards_plot(self.accumulated_rewards)
                except Exception as e:
                    print(f"ERROR: 更新奖励曲线时出错: {str(e)}")

            # 更新收益曲线
            if returns_updated and hasattr(self, 'returns_history') and self.returns_history:
                try:
                    self.update_returns_plot(self.returns_history)
                except Exception as e:
                    print(f"ERROR: 更新收益曲线时出错: {str(e)}")

            # 更新学习率曲线
            if learning_rate_updated and hasattr(self, 'learning_rates_history') and self.learning_rates_history:
                try:
                    self.update_learning_rate_plot(self.learning_rates_history)
                except Exception as e:
                    print(f"ERROR: 更新学习率曲线时出错: {str(e)}")

            # 在所有图表更新后，一次性触发UI事件处理
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

    def handle_evaluation_result(self, data):
        """处理评估结果数据"""
        # 更新图表和指标
        if 'portfolio_values' in data:
            self.plot_portfolio_curve(data['portfolio_values'])

        if 'metrics' in data:
            self.update_metrics_text(data['metrics'])

        # 更新状态
        self.status_label.setText("评估完成")

    def update_training_trade_table(self, trades):
        """更新训练交易表格"""
        try:
            if not trades:
                self.log_message("没有训练交易记录可显示")
                return

            print(f"DEBUG: 更新训练交易表格, 共 {len(trades)} 条记录")

            # 清空表格
            self.training_trades_table.setRowCount(0)

            # 添加交易记录
            for row, trade in enumerate(trades):
                self.training_trades_table.insertRow(row)

                # 获取时间
                timestamp = trade.get('timestamp', trade.get('time', ''))
                self.training_trades_table.setItem(row, 0, QTableWidgetItem(str(timestamp)))

                # 设置操作（带颜色）
                action_text = trade.get('action', trade.get('type', '未知'))
                # 标准化操作名称
                if action_text == 'buy':
                    action_text = '买入'
                elif action_text == 'sell':
                    action_text = '卖出'

                action_item = QTableWidgetItem(action_text)
                if action_text == '买入':
                    action_item.setForeground(QColor('green'))
                elif action_text == '卖出':
                    action_item.setForeground(QColor('red'))
                self.training_trades_table.setItem(row, 1, action_item)

                # 获取必要数据
                price = trade.get('price', 0)
                amount = trade.get('amount', 0)

                # 设置价格
                self.training_trades_table.setItem(row, 2, QTableWidgetItem(f"{price:.2f}"))

                # 设置数量
                self.training_trades_table.setItem(row, 3, QTableWidgetItem(f"{amount:.6f}"))

                # 获取交易费率
                transaction_fee = 0.0005  # 默认值
                if hasattr(self, 'fee_spin'):
                    transaction_fee = self.fee_spin.value() / 100  # 从UI获取

                # 设置交易金额
                self.process_trade_amount(self.training_trades_table, row, 4, trade, action_text, price, amount, transaction_fee)

                # 设置手续费
                self.process_trade_fee(self.training_trades_table, row, 5, trade, action_text, price, amount, transaction_fee)

                # 设置收益和收益率
                self.process_trade_profit(self.training_trades_table, row, 6, 7, trade, action_text, price, amount, transaction_fee)

                # 设置余额和总资产
                balance = trade.get('balance', 0)
                total_value = trade.get('total_value', trade.get('portfolio_value', 0))
                self.training_trades_table.setItem(row, 8, QTableWidgetItem(f"{balance:.2f}"))
                self.training_trades_table.setItem(row, 9, QTableWidgetItem(f"{total_value:.2f}"))

            # 调整列宽
            self.training_trades_table.resizeColumnsToContents()

            # 不再自动切换到训练交易记录标签页

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"ERROR: 更新训练交易表格时出错: {str(e)}\n{error_msg}")
            self.log_message(f"更新训练交易表格时出错: {str(e)}")

    def update_evaluation_trade_table(self, trades):
        """更新评估交易表格"""
        try:
            if not trades:
                self.log_message("没有评估交易记录可显示")
                return

            print(f"DEBUG: 更新评估交易表格, 共 {len(trades)} 条记录")

            # 清空表格
            self.evaluation_trades_table.setRowCount(0)

            # 添加交易记录
            for row, trade in enumerate(trades):
                self.evaluation_trades_table.insertRow(row)

                # 获取时间
                timestamp = trade.get('timestamp', trade.get('time', ''))
                self.evaluation_trades_table.setItem(row, 0, QTableWidgetItem(str(timestamp)))

                # 设置操作（带颜色）
                action_text = trade.get('action', trade.get('type', '未知'))
                # 标准化操作名称
                if action_text == 'buy':
                    action_text = '买入'
                elif action_text == 'sell':
                    action_text = '卖出'

                action_item = QTableWidgetItem(action_text)
                if action_text == '买入':
                    action_item.setForeground(QColor('green'))
                elif action_text == '卖出':
                    action_item.setForeground(QColor('red'))
                self.evaluation_trades_table.setItem(row, 1, action_item)

                # 获取必要数据
                price = trade.get('price', 0)
                amount = trade.get('amount', 0)

                # 设置价格
                self.evaluation_trades_table.setItem(row, 2, QTableWidgetItem(f"{price:.2f}"))

                # 设置数量
                self.evaluation_trades_table.setItem(row, 3, QTableWidgetItem(f"{amount:.6f}"))

                # 获取交易费率
                transaction_fee = 0.0005  # 默认值
                if hasattr(self, 'fee_spin'):
                    transaction_fee = self.fee_spin.value() / 100  # 从UI获取

                # 设置交易金额
                self.process_trade_amount(self.evaluation_trades_table, row, 4, trade, action_text, price, amount, transaction_fee)

                # 设置手续费
                self.process_trade_fee(self.evaluation_trades_table, row, 5, trade, action_text, price, amount, transaction_fee)

                # 设置收益和收益率
                self.process_trade_profit(self.evaluation_trades_table, row, 6, 7, trade, action_text, price, amount, transaction_fee)

                # 设置余额和总资产
                balance = trade.get('balance', 0)
                total_value = trade.get('total_value', trade.get('portfolio_value', 0))
                self.evaluation_trades_table.setItem(row, 8, QTableWidgetItem(f"{balance:.2f}"))
                self.evaluation_trades_table.setItem(row, 9, QTableWidgetItem(f"{total_value:.2f}"))

            # 调整列宽
            self.evaluation_trades_table.resizeColumnsToContents()

            # 不再自动切换到评估交易记录标签页

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"ERROR: 更新评估交易表格时出错: {str(e)}\n{error_msg}")
            self.log_message(f"更新评估交易表格时出错: {str(e)}")

    def process_trade_amount(self, table, row, col, trade, action_text, price, amount, transaction_fee):
        """处理交易金额"""
        if action_text == '买入':
            # 使用'buy_amount'字段或计算
            if 'buy_amount' in trade:
                trade_amount = trade['buy_amount']
            else:
                trade_amount = amount * price
            table.setItem(row, col, QTableWidgetItem(f"{trade_amount:.2f}"))
        elif action_text == '卖出':
            # 使用'sell_value'字段或计算
            if 'sell_value' in trade:
                trade_amount = trade['sell_value']
            else:
                trade_amount = amount * price * (1 - transaction_fee)
            table.setItem(row, col, QTableWidgetItem(f"{trade_amount:.2f}"))
        else:
            # 未知操作
            table.setItem(row, col, QTableWidgetItem("0.00"))

    def process_trade_fee(self, table, row, col, trade, action_text, price, amount, transaction_fee):
        """处理交易手续费"""
        if action_text == '买入':
            # 使用'buy_fee'字段或计算
            if 'buy_fee' in trade:
                fee = trade['buy_fee']
            else:
                fee = amount * price * transaction_fee
            table.setItem(row, col, QTableWidgetItem(f"{fee:.2f}"))
        elif action_text == '卖出':
            # 使用'sell_fee'字段或计算
            if 'sell_fee' in trade:
                fee = trade['sell_fee']
            else:
                fee = amount * price * transaction_fee
            table.setItem(row, col, QTableWidgetItem(f"{fee:.2f}"))
        else:
            # 未知操作
            table.setItem(row, col, QTableWidgetItem("0.00"))

    def process_trade_profit(self, table, row, profit_col, profit_pct_col, trade, action_text, price, amount, transaction_fee):
        """处理交易收益和收益率"""
        # 设置收益（带颜色）
        profit = trade.get('profit', 0)
        # 如果记录中没有profit字段，尝试计算
        if (profit == 0 or profit is None) and action_text == '卖出' and 'last_buy_price' in trade:
            last_buy_price = trade['last_buy_price']
            sell_price = price
            # 计算收益，考虑手续费
            buy_cost = amount * last_buy_price * (1 + transaction_fee)
            sell_income = amount * sell_price * (1 - transaction_fee)
            profit = sell_income - buy_cost

        profit_item = QTableWidgetItem(f"{profit:.2f}")
        if profit > 0:
            profit_item.setForeground(QColor('green'))
        elif profit < 0:
            profit_item.setForeground(QColor('red'))
        table.setItem(row, profit_col, profit_item)

        # 设置收益率（带颜色）
        profit_pct = trade.get('profit_pct', 0)
        # 如果记录中没有profit_pct字段，尝试计算
        if (profit_pct == 0 or profit_pct is None) and action_text == '卖出' and 'last_buy_price' in trade:
            last_buy_price = trade['last_buy_price']
            sell_price = price
            # 计算收益率，不考虑手续费的简单百分比
            profit_pct = (sell_price - last_buy_price) / last_buy_price * 100

        profit_pct_item = QTableWidgetItem(f"{profit_pct:.2f}")
        if profit_pct > 0:
            profit_pct_item.setForeground(QColor('green'))
        elif profit_pct < 0:
            profit_pct_item.setForeground(QColor('red'))
        table.setItem(row, profit_pct_col, profit_pct_item)

    def on_training_update(self, data):
        """处理训练进度更新"""
        try:
            print(f"DEBUG - on_training_update: 收到数据 keys={list(data.keys())}")

            # 判断数据来源类型
            source_type = data.get('source_type', 'training')

            # 根据不同来源类型处理数据
            if source_type == 'training':
                # 处理训练数据
                self.handle_training_progress(data)

                # 查找交易记录 - 尝试所有可能的键名
                trade_data = None
                if 'trade_records' in data and data['trade_records']:
                    trade_data = data['trade_records']
                    print(f"DEBUG - on_training_update: 找到trade_records, 长度={len(trade_data)}")
                elif 'trades' in data and data['trades']:
                    trade_data = data['trades']
                    print(f"DEBUG - on_training_update: 找到trades, 长度={len(trade_data)}")

                # 更新交易表格
                if trade_data:
                    self.log_message(f"当前回合有 {len(trade_data)} 条交易记录")
                    if not hasattr(self, 'training_trades') or not self.training_trades:
                        self.training_trades = []

                    # 添加新的交易记录
                    self.training_trades = trade_data

                    # 更新交易表格
                    self.update_training_trade_table(self.training_trades)

                    # 直接从交易数据更新收益曲线和奖励曲线
                    self.update_training_charts_from_trades(trade_data)

            elif source_type == 'evaluation':
                # 处理评估数据
                self.handle_evaluation_result(data)

                # 查找交易记录
                trade_data = None
                if 'trade_history' in data and data['trade_history']:
                    trade_data = data['trade_history']
                    print(f"DEBUG - on_training_update: 找到evaluation trade_history, 长度={len(trade_data)}")
                elif 'trades' in data and data['trades']:
                    trade_data = data['trades']
                    print(f"DEBUG - on_training_update: 找到evaluation trades, 长度={len(trade_data)}")

                # 更新评估交易表格
                if trade_data:
                    self.log_message(f"评估回合有 {len(trade_data)} 条交易记录")
                    if not hasattr(self, 'evaluation_trades') or not self.evaluation_trades:
                        self.evaluation_trades = []

                    # 添加新的交易记录
                    self.evaluation_trades = trade_data

                    # 更新交易表格
                    self.update_evaluation_trade_table(self.evaluation_trades)

                    # 直接从评估交易数据更新图表
                    self.update_evaluation_charts_from_trades(trade_data)

            # 更新环境信息和资产价值
            if 'env_info' in data:
                env_info = data['env_info']
                step = env_info.get('current_step', 0)
                max_steps = env_info.get('max_episode_steps', 0)
                portfolio_value = env_info.get('balance', 0) + env_info.get('position_value', 0)

                if max_steps > 0:
                    self.step_label.setText(f"步数: {step}/{max_steps} - 资产: {portfolio_value:.2f}")

            # 如果标记为最终更新，重置状态
            if data.get('final', False):
                self.on_training_completed()

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"ERROR - on_training_update: 处理训练更新时出错: {str(e)}\n{error_msg}")
            self.log_message(f"处理训练更新时出错: {str(e)}")

    def update_training_charts_from_trades(self, trades):
        """直接从交易数据更新训练图表"""
        try:
            if not trades:
                return

            # 收集资产曲线数据
            timestamps = []
            portfolio_values = []
            rewards = []
            steps = []

            # 从交易记录中提取数据
            for trade in trades:
                timestamps.append(trade.get('timestamp', trade.get('time', '')))
                portfolio_value = trade.get('total_value', trade.get('portfolio_value', 0))
                portfolio_values.append(portfolio_value)
                steps.append(trade.get('step', len(steps)))  # 获取训练步数，如果没有则使用序号
                if 'reward' in trade:
                    rewards.append(trade.get('reward', 0))

            # 更新收益曲线
            if portfolio_values:
                print(f"DEBUG: 直接从数据源更新训练收益曲线, 数据点数: {len(portfolio_values)}")
                self.returns_steps = steps  # 保存步数数据
                self.update_returns_plot(portfolio_values)

                # 如果数据中有奖励信息，更新奖励曲线
                if rewards:
                    print(f"DEBUG: 直接从数据源更新训练奖励曲线, 数据点数: {len(rewards)}")
                    self.update_rewards_plot(rewards)

                # 不再自动切换标签页，让用户主导UI交互

        except Exception as e:
            print(f"ERROR: 从交易数据更新训练图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_evaluation_charts_from_trades(self, trades):
        """直接从交易数据更新评估图表"""
        try:
            if not trades:
                return

            # 收集资产曲线数据
            timestamps = []
            portfolio_values = []

            # 从交易记录中提取数据
            for trade in trades:
                timestamps.append(trade.get('timestamp', trade.get('time', '')))
                portfolio_value = trade.get('total_value', trade.get('portfolio_value', 0))
                portfolio_values.append(portfolio_value)

            # 更新收益曲线
            if portfolio_values:
                print(f"DEBUG: 直接从数据源更新评估资产曲线, 数据点数: {len(portfolio_values)}")
                self.plot_portfolio_curve(portfolio_values)

                # 不再自动切换标签页，保持用户当前选择的标签页

        except Exception as e:
            print(f"ERROR: 从交易数据更新评估图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_eval_result(self, data: Dict):
        """处理评估结果"""
        try:
            print(f"DEBUG - on_eval_result: 收到评估数据 keys={list(data.keys())}")

            # 更新图表和指标
            if 'portfolio_values' in data:
                print(f"DEBUG - on_eval_result: 绘制资产曲线, 长度={len(data['portfolio_values'])}")
                self.plot_portfolio_curve(data['portfolio_values'])

            if 'metrics' in data:
                print(f"DEBUG - on_eval_result: 更新指标, metrics keys={list(data['metrics'].keys())}")
                self.update_metrics_text(data['metrics'])

            # 查找交易记录 - 尝试所有可能的键名
            trade_data = None
            if 'trade_history' in data and data['trade_history']:
                trade_data = data['trade_history']
                print(f"DEBUG - on_eval_result: 找到trade_history, 长度={len(trade_data)}")
            elif 'trades' in data and data['trades']:
                trade_data = data['trades']
                print(f"DEBUG - on_eval_result: 找到trades, 长度={len(trade_data)}")
            else:
                # 如果在data中没有直接找到，检查是否有嵌套的metrics字典
                if 'metrics' in data and isinstance(data['metrics'], dict):
                    metrics = data['metrics']
                    if 'trade_history' in metrics and metrics['trade_history']:
                        trade_data = metrics['trade_history']
                        print(f"DEBUG - on_eval_result: 在metrics中找到trade_history, 长度={len(trade_data)}")
                    elif 'trades' in metrics and metrics['trades']:
                        trade_data = metrics['trades']
                        print(f"DEBUG - on_eval_result: 在metrics中找到trades, 长度={len(trade_data)}")

            # 如果找到交易数据，处理它
            if trade_data:
                self.log_message(f"收到 {len(trade_data)} 条交易记录")

                # 1. 更新评估交易表格
                self.update_evaluation_trade_table(trade_data)

                # 2. 直接从交易数据更新图表
                self.update_evaluation_charts_from_trades(trade_data)

                # 3. 从交易数据中提取收益率和资产曲线
                if not 'portfolio_values' in data:
                    portfolio_values = []
                    for trade in trade_data:
                        portfolio_value = trade.get('total_value', trade.get('portfolio_value', 0))
                        if portfolio_value:
                            portfolio_values.append(portfolio_value)

                    if portfolio_values:
                        print(f"DEBUG - on_eval_result: 从交易记录提取的资产曲线, 长度={len(portfolio_values)}")
                        self.plot_portfolio_curve(portfolio_values)
            else:
                self.log_message("没有收到交易记录数据")
                print("DEBUG - on_eval_result: 未找到任何交易记录数据")

            # 更新状态
            self.status_label.setText("评估完成")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"ERROR - on_eval_result: 处理评估结果时出错: {str(e)}\n{error_msg}")
            self.log_message(f"处理评估结果时出错: {str(e)}")
            traceback.print_exc()

    def evaluate_best_model(self):
        """评估最佳模型并显示结果"""
        if not hasattr(self, 'best_model') or self.best_model is None:
            QMessageBox.warning(self, "警告", "没有可用的最佳模型，请先训练模型")
            return

        self.log_message("开始评估最佳模型...")
        print("DEBUG: 开始评估最佳模型...")

        # 使用验证集数据进行评估
        train_size = int(len(self.kline_data) * self.train_ratio_spin.value() / 100)
        val_data = self.kline_data.iloc[train_size:].copy()

        # 创建环境配置
        config_data = self.get_env_config()
        env_config = config_data['env_config']
        reward_weights = config_data.get('reward_weights', {})
        reward_config = config_data.get('reward_config', {})

        # 创建评估环境
        from rl_strategies.environments.trading_env import TradingEnv
        eval_env = TradingEnv(
            df=val_data,
            **env_config  # 直接传递环境配置
        )

        # 如果使用复合奖励，设置奖励权重
        if env_config['reward_type'] == 'compound':
            eval_env.reward_weights = reward_weights
            eval_env.reward_config = reward_config

        # 重置环境
        state, _ = eval_env.reset()
        done = False

        # 确保交易历史已初始化
        if not hasattr(eval_env, 'trade_history') or eval_env.trade_history is None:
            eval_env.trade_history = []

        # 记录评估数据
        portfolio_values = [env_config['initial_balance']]
        timestamp_values = []  # 用于存储时间戳
        trade_times = []
        trade_prices = []
        trade_actions = []
        trade_portfolio_values = []

        # 运行评估
        print(f"DEBUG: 开始评估最佳模型，数据长度: {len(val_data)}")
        while not done:
            # 使用最佳模型选择动作
            action = self.best_model.act(state)

            # 执行动作
            next_state, _, done, _, info = eval_env.step(action)

            # 记录数据
            portfolio_values.append(info['portfolio_value'])

            # 记录时间戳，如果可用
            if hasattr(eval_env, 'df') and hasattr(eval_env, 'current_step'):
                try:
                    timestamp = eval_env.df.index[eval_env.current_step]
                    timestamp_values.append(timestamp)
                except:
                    timestamp_values.append(len(portfolio_values) - 1)
            else:
                timestamp_values.append(len(portfolio_values) - 1)

            # 如果有交易发生，记录交易数据
            if len(eval_env.trade_history) > 0 and eval_env.trade_history[-1]['timestamp'] == eval_env.df.iloc[eval_env.current_step].name:
                trade = eval_env.trade_history[-1]
                trade_times.append(len(portfolio_values) - 1)  # 交易发生的步数
                trade_prices.append(trade['price'])
                trade_actions.append(trade['action'])
                trade_portfolio_values.append(info['portfolio_value'])  # 使用实际资金金额

            # 更新状态
            state = next_state

        # 计算性能指标
        initial_balance = env_config['initial_balance']
        final_value = portfolio_values[-1]
        absolute_profit = final_value - initial_balance
        total_return_pct = (absolute_profit / initial_balance * 100) if initial_balance > 0 else 0

        # 计算最大回撤
        max_drawdown = eval_env.max_drawdown * 100

        # 计算夏普比率
        returns_pct = []
        for i in range(1, len(portfolio_values)):
            pct = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] * 100
            returns_pct.append(pct)

        if len(returns_pct) > 1:
            returns_array = np.array(returns_pct)
            returns_mean = np.mean(returns_array)
            returns_std = np.std(returns_array)
            sharpe_ratio = returns_mean / (returns_std + 1e-9)
        else:
            sharpe_ratio = 0

        # 计算胜率
        win_trades = sum(1 for trade in eval_env.trade_history if trade['action'] == '卖出' and trade.get('profit', 0) > 0)
        total_trades = sum(1 for trade in eval_env.trade_history if trade['action'] == '卖出')
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

        # 更新性能指标显示
        metrics_text = (
            f"初始资金: {initial_balance:.2f}\n"
            f"最终价值: {final_value:.2f}\n"
            f"绝对收益: {absolute_profit:.2f}\n"
            f"收益率: {total_return_pct:.2f}%\n"
            f"最大回撤: {max_drawdown:.2f}%\n"
            f"夏普比率: {sharpe_ratio:.4f}\n"
            f"总交易次数: {total_trades}\n"
            f"获利交易: {win_trades}\n"
            f"胜率: {win_rate:.2f}%\n"
        )

        self.best_metrics_text.setText(metrics_text)

        # 更新最佳模型信息
        self.best_model_info.setText(f"最佳模型 (回合 {self.best_model_episode}) | 收益: {absolute_profit:.2f} ({total_return_pct:.2f}%)")

        # 绘制收益曲线（使用实际资金金额）
        self.best_model_ax.clear()
        self.best_model_ax.plot(portfolio_values, 'b-')
        self.best_model_ax.set_title('最佳模型投资组合价值曲线')
        self.best_model_ax.set_xlabel('交易步数')
        self.best_model_ax.set_ylabel('资金金额')
        self.best_model_ax.grid(True)

        # 添加初始资金线
        self.best_model_ax.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.7, label='初始资金')

        # 标记交易点
        if trade_times:
            buy_times = [t for t, a in zip(trade_times, trade_actions) if a == '买入']
            buy_values = [v for v, a in zip(trade_portfolio_values, trade_actions) if a == '买入']

            sell_times = [t for t, a in zip(trade_times, trade_actions) if a == '卖出']
            sell_values = [v for v, a in zip(trade_portfolio_values, trade_actions) if a == '卖出']

            if buy_times:
                self.best_model_ax.scatter(buy_times, buy_values, color='green', marker='^', s=100, label='买入')

            if sell_times:
                self.best_model_ax.scatter(sell_times, sell_values, color='red', marker='v', s=100, label='卖出')

        # 设置Y轴范围，确保初始资金线在图表中间位置
        min_val = min(portfolio_values) if portfolio_values else initial_balance * 0.8
        max_val = max(portfolio_values) if portfolio_values else initial_balance * 1.2

        # 确保Y轴有足够的空间
        y_range = max(max_val - min_val, initial_balance * 0.4)  # 至少显示初始资金的40%范围
        self.best_model_ax.set_ylim(
            min(min_val, initial_balance - y_range * 0.2),  # 下限
            max(max_val, initial_balance + y_range * 0.8)   # 上限
        )

        self.best_model_ax.legend()
        self.best_model_figure.tight_layout()
        self.best_model_canvas.draw()

        # 直接输出交易历史记录以便调试
        print(f"DEBUG: 评估完成，获取到 {len(eval_env.trade_history)} 条交易记录")
        for i, trade in enumerate(eval_env.trade_history[:5]):
            print(f"DEBUG: 交易记录 {i+1}: {trade}")

        # 更新交易表格，使用通用的更新方法
        self.log_message(f"评估得到 {len(eval_env.trade_history)} 条交易记录")
        self.update_trade_table(eval_env.trade_history)

        # 将评估结果保存到文件以便分析
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_model_eval_{timestamp}.json"

            # 自定义JSON序列化器处理特殊类型
            def json_serial(obj):
                """处理无法序列化的对象"""
                if hasattr(obj, 'isoformat'):  # 处理datetime和Timestamp对象
                    return obj.isoformat()
                elif hasattr(obj, 'item'):  # 处理numpy数值类型
                    return obj.item()
                elif isinstance(obj, np.ndarray):  # 处理numpy数组
                    return obj.tolist()
                return str(obj)  # 对于其他无法序列化的对象转为字符串

            # 准备可序列化的数据
            save_data = {
                "metrics": {
                    "initial_balance": float(initial_balance),
                    "final_value": float(final_value),
                    "absolute_profit": float(absolute_profit),
                    "total_return_pct": float(total_return_pct),
                    "max_drawdown": float(max_drawdown),
                    "sharpe_ratio": float(sharpe_ratio),
                    "total_trades": int(total_trades),
                    "win_trades": int(win_trades),
                    "win_rate": float(win_rate)
                },
                "portfolio_values": [float(v) for v in portfolio_values],
                "trade_history": eval_env.trade_history
            }

            import json
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2, default=json_serial)
            print(f"DEBUG: 已保存评估结果到 {filename}")
        except Exception as e:
            print(f"ERROR: 保存评估结果失败: {str(e)}")

        # 切换到最佳模型评估标签页
        self.results_tabs.setCurrentIndex(2)  # 最佳模型评估标签页

        self.log_message(f"最佳模型评估完成，绝对收益: {absolute_profit:.2f}，收益率: {total_return_pct:.2f}%，胜率: {win_rate:.2f}%")

    def get_env_config(self):
        """获取环境配置参数"""
        print("DEBUG: 正在获取环境配置")

        # 获取当前所有环境参数
        env_config = {
            'window_size': int(self.window_spin.value()),
            'initial_balance': float(self.balance_spin.value()),
            'transaction_fee': float(self.fee_spin.value()) / 100.0,  # 从百分比转为小数
            'reward_type': self.reward_type_combo.currentText(),
            'use_technical_indicators': self.use_indicators_check.isChecked(),
            'include_position': self.include_position_check.isChecked(),
            'penalize_inaction': self.penalize_inaction_check.isChecked(),
            'max_position_size': float(self.max_position_spin.value()) / 100.0,  # 从百分比转为小数
            'base_position_size': float(self.base_position_spin.value()) / 100.0,  # 从百分比转为小数
            'position_sizing': self.position_sizing_check.isChecked(),
            'fixed_trade_amount': float(self.fixed_amount_spin.value()),
            'max_trade_amount': float(self.max_trade_amount_spin.value()) / 100.0,  # 从百分比转为小数
            'max_episode_steps': int(self.max_steps_spin.value()),  # 注意改为max_episode_steps
            'early_stop_loss_threshold': float(self.stop_loss_spin.value()) / 100.0,  # 从百分比转为小数
            'early_stop_enabled': self.early_stop_check.isChecked(),
            'min_trade_interval': int(self.min_interval_spin.value()),
            'max_trades_per_episode': int(self.max_trades_spin.value())
        }

        # 准备配置字典
        config = {
            'env_config': env_config,
            'reward_weights': {},
            'reward_config': {}
        }

        # 添加复合奖励权重
        if env_config['reward_type'] == '复合奖励':
            config['reward_weights'] = {
                'profit': float(self.profit_weight_spin.value()),
                'cumulative_return': float(self.cum_return_weight_spin.value()),
                'risk_adjusted': float(self.risk_weight_spin.value()),
                'drawdown': float(self.drawdown_weight_spin.value()),
                'trade_frequency': float(self.freq_weight_spin.value()),
                'inaction': float(self.inaction_spin.value()),
                'trend_follow': float(self.trend_weight_spin.value()),
                'consecutive_buy': float(self.consecutive_buy_spin.value())
            }

            # 添加高级奖励配置
            config['reward_config'] = {
                'max_reward_limit': float(self.max_reward_spin.value()),
                'max_drawdown_penalty': float(self.max_drawdown_spin.value()),
                'inaction_base_penalty': float(self.inaction_spin.value()),  # 使用正确的变量名
                'inaction_time_penalty': float(self.inaction_spin.value()),  # 修正变量名
                'trend_misalign_penalty': float(self.trend_misalign_spin.value()),
                'trend_follow_reward': float(self.trend_follow_spin.value()),
                'frequent_trade_penalty': float(self.frequent_trade_spin.value()),
                'position_holding_penalty': float(self.position_holding_spin.value()),
                'consecutive_buy_base_penalty': float(self.consecutive_buy_spin.value()),
                'trade_interval_threshold': int(self.trade_interval_spin.value()),
                'profit_base_reward': float(self.profit_base_reward_spin.value()),  # 新增：成功交易基础奖励
                'reward_amplifier': float(self.reward_amplifier_spin.value())       # 新增：奖励放大因子
            }

        # 确保max_episode_steps有一个合理的值
        if env_config['max_episode_steps'] <= 0:
            env_config['max_episode_steps'] = 1000  # 默认给一个合理的值
            print(f"警告: max_episode_steps被设置为非正值，已自动调整为 {env_config['max_episode_steps']}")

        # 确保步数是合理的，不要太小
        min_allowed_steps = 20
        if env_config['max_episode_steps'] < min_allowed_steps:
            env_config['max_episode_steps'] = min_allowed_steps
            print(f"警告: max_episode_steps太小，已自动调整为至少 {min_allowed_steps} 步")

        print(f"DEBUG: 环境配置参数: {env_config}")
        print(f"DEBUG: 奖励权重: {config['reward_weights']}")
        print(f"DEBUG: 奖励配置: {config['reward_config']}")

        return config

    def on_reward_type_changed(self, index):
        """
        当奖励类型改变时触发
        处理复合奖励设置的可见性
        """
        # 只有选择"复合奖励"时才显示复合奖励权重设置
        is_compound = index == 2  # 复合奖励的索引是2
        self.compound_reward_group.setVisible(is_compound)

        # 更新高级奖励配置组的可见性
        self.advanced_reward_group.setVisible(is_compound)  # 复合奖励时显示高级配置

        # 在奖励类型改变时，同步修改get_env_config方法中的默认值
        reward_type_mapping = {
            0: 'profit',    # 利润
            1: 'sharpe',    # 夏普比率
            2: 'compound'   # 复合奖励
        }
        reward_type = reward_type_mapping.get(index, 'compound')
        print(f"DEBUG: 奖励类型改变为: {reward_type}")

        # 打印调试信息
        print(f"DEBUG: 连续买入惩罚权重: {self.consecutive_buy_weight_spin.value()}")
        print(f"DEBUG: 连续买入基础惩罚: {self.consecutive_buy_base_spin.value()}")

    def on_model_changed(self, index):
        """
        当模型类型改变时触发
        更新与模型相关的UI元素状态
        """
        model_type = self.model_type_combo.currentText()
        print(f"DEBUG: 模型类型改变为: {model_type}")

        # 根据模型类型更新相关UI元素状态
        is_dqn = model_type == "DQN"
        self.double_dqn_check.setVisible(is_dqn)  # 只有DQN才显示Double DQN选项

        # 根据不同模型类型设置合适的默认参数
        if model_type == "DQN":
            # DQN默认参数
            self.lr_spin.setValue(0.0005)
            self.gamma_spin.setValue(0.99)
            self.batch_size_spin.setValue(64)
        elif model_type == "PPO":
            # PPO默认参数
            self.lr_spin.setValue(0.0003)
            self.gamma_spin.setValue(0.99)
            self.batch_size_spin.setValue(128)
        elif model_type == "A2C":
            # A2C默认参数
            self.lr_spin.setValue(0.0007)
            self.gamma_spin.setValue(0.99)
            self.batch_size_spin.setValue(32)
        elif model_type == "DDPG":
            # DDPG默认参数
            self.lr_spin.setValue(0.0001)
            self.gamma_spin.setValue(0.98)
            self.batch_size_spin.setValue(64)

        self.log_message(f"已切换到{model_type}模型类型")

    def log_message(self, message):
        """
        向日志文本框添加消息

        参数:
            message: 要添加的消息
        """
        # 获取当前时间
        current_time = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{current_time}] {message}"

        # 在主日志文本框中添加带时间戳的消息
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(formatted_message)
            # 自动滚动到底部
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        # 同时打印到控制台，便于调试
        print(f"LOG: {message}")

    def start_training(self):
        """开始训练"""
        # 检查是否已经在训练状态
        if self.is_training:
            self.log_message("训练已经在进行中...")
            return
        
        try:
            # 在开始新的训练前重置历史数据和图表
            self._reset_history_data()
            
            # 检查配置
            if hasattr(self, 'model_type_combo'):
                model_type = self.model_type_combo.currentText()
            else:
                model_type = "DQN"
            print(f"选择的模型类型: {model_type}")
            
            # 设置环境和奖励类型
            if hasattr(self, 'env_combo'):
                env_name = self.env_combo.currentText()
            else:
                env_name = "TradingEnv"
            print(f"选择的环境: {env_name}")
            
            # 重置UI侧步数计数器
            self.ui_step_counter = 0
            print("[训练开始] 重置UI步数计数器为0")
            
            # 重置UI侧探索率变量
            self.ui_epsilon_value = 1.0
            print("[训练开始] 重置UI探索率为1.0")
            
            # 输出训练前的配置信息
            print("\n" + "="*50)
            print("训练开始，配置信息如下:")

            # 获取环境配置
            try:
                env_config = self.get_env_config()
                print(f"[训练] 环境配置: {env_config}")
            except Exception as e:
                QMessageBox.critical(self, "配置错误", f"环境配置错误: {str(e)}")
                return

            # 获取代理配置
            try:
                agent_config = self.get_agent_config()
                print("\n=== 训练开始时的代理配置 ===")
                print(f"[训练] 基础学习率: {agent_config['learning_rate']:.6f}")
                print(f"[训练] 最大学习率(max_learning_rate): {agent_config['max_learning_rate']:.6f}")
                if 'lr_adaptation' in agent_config:
                    print(f"[训练] 学习率自适应最大学习率: {agent_config['lr_adaptation']['max_lr']:.6f}")
                print("="*50 + "\n")
            except Exception as e:
                QMessageBox.critical(self, "配置错误", f"代理配置错误: {str(e)}")
                return

            # 获取训练参数
            self.log_message("准备训练参数...")

            # 获取模型参数
            use_double_dqn = self.double_dqn_check.isChecked()
            hidden_layers_text = self.hidden_layers_text.currentText()
            hidden_layers = [int(x) for x in hidden_layers_text.split(',')]
            learning_rate = self.lr_spin.value()
            gamma = self.gamma_spin.value()
            batch_size = self.batch_size_spin.value()
            eval_freq = self.eval_freq_spin.value()

            # 获取训练参数
            max_episodes = self.max_episodes_spin.value()
            train_ratio = self.train_ratio_spin.value() / 100.0  # 转换为小数

            # 获取环境配置
            config_data = self.get_env_config()  # 获取包含多个配置的字典
            env_config = config_data['env_config']  # 环境基本配置
            reward_weights = config_data.get('reward_weights', {})  # 奖励权重
            reward_config = config_data.get('reward_config', {})  # 奖励配置

            # 如果使用复合奖励，将奖励权重添加到环境配置中
            if env_config['reward_type'] == 'compound':
                env_config['reward_weights'] = reward_weights

            # 分割训练和验证数据
            train_size = int(len(self.kline_data) * train_ratio)
            train_data = self.kline_data.iloc[:train_size].copy()
            val_data = self.kline_data.iloc[train_size:].copy()

            self.log_message(f"训练数据: {len(train_data)}行, 验证数据: {len(val_data)}行")

            # 创建代理配置
            agent_config = {
                'learning_rate': learning_rate,
                'gamma': gamma,
                'batch_size': batch_size,
                'hidden_layers': hidden_layers,
                'use_double_dqn': use_double_dqn,
                'eval_frequency': eval_freq,
                'max_episodes': max_episodes
            }

            # 创建训练配置
            train_config = {
                'episodes': max_episodes,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'discount_factor': gamma,
                'agent_type': model_type.lower(),
                'verbose': True,
                'max_learning_rate': self.model_select_max_lr_spin.value(),  # 直接从UI控件获取
                'lr_adaptation': {  # 添加学习率自适应配置
                    'enabled': hasattr(self, 'dynamic_lr_check') and self.dynamic_lr_check.isChecked(),
                    'base_lr': learning_rate,
                    'min_lr': self.min_lr_spin.value() if hasattr(self, 'min_lr_spin') else learning_rate / 10.0,
                    'max_lr': self.model_select_max_lr_spin.value(),  # 使用相同的最大学习率
                    'increase_threshold': self.lr_increase_threshold_spin.value() if hasattr(self, 'lr_increase_threshold_spin') else -0.1,
                    'decrease_threshold': self.lr_decrease_threshold_spin.value() if hasattr(self, 'lr_decrease_threshold_spin') else 0.05,
                    'increase_factor': self.lr_increase_factor_spin.value() if hasattr(self, 'lr_increase_factor_spin') else 1.5,
                    'decrease_factor': self.lr_decrease_factor_spin.value() if hasattr(self, 'lr_decrease_factor_spin') else 0.8,
                    'adaptation_window': 5,
                    'cooldown_period': 3
                }
            }

            # 打印配置信息以便调试
            print("\n=== 训练配置信息 ===")
            print(f"基础学习率: {learning_rate}")
            print(f"最大学习率: {train_config['max_learning_rate']}")
            print(f"学习率自适应配置: {train_config['lr_adaptation']}")
            print("="*50)

            # 根据模型类型设置正确的agent_type参数
            agent_type = model_type.lower()

            try:
                # 直接创建训练线程，而不是先创建训练器
                from rl_strategies.rl_training_thread import RLTrainingThread
                self.training_thread = RLTrainingThread(
                    trainer=None,  # 不使用预先创建的训练器
                    max_episodes=max_episodes,
                    env_config=env_config,
                    train_config=train_config,  # 包含完整的配置
                    train_df=train_data,
                    eval_df=val_data,
                    load_model_path=None,
                    save_model_path=None
                )

                # 连接信号
                self.training_thread.progress_signal.connect(self.on_training_update)
                self.training_thread.eval_signal.connect(self.on_eval_result)
                self.training_thread.complete_signal.connect(self.on_training_completed)
                self.training_thread.error_signal.connect(self.log_message)  # 连接错误信号
                self.training_thread.log_signal.connect(self.log_message)
                
                # 连接探索率信号
                if hasattr(self.training_thread, 'epsilon_signal'):
                    self.training_thread.epsilon_signal.connect(self.update_epsilon_info)
                    print("已连接探索率信号到UI")
                
                # 设置探索率回调函数
                # 注意：此时trainer和agent可能还未创建，需要在训练线程中设置回调
                # 添加自定义回调函数到训练线程，让它在创建agent后设置回调
                def setup_epsilon_callback(agent):
                    if hasattr(agent, 'register_epsilon_callback'):
                        agent.register_epsilon_callback(self.update_epsilon_info)
                        print("已注册探索率变化回调函数")
                    elif hasattr(agent, 'epsilon_callback'):
                        agent.epsilon_callback = self.update_epsilon_info
                        print("已设置探索率回调函数")
                    else:
                        print("警告: 代理对象不支持探索率回调")
                
                # 将回调设置函数传递给训练线程
                self.training_thread.setup_agent_callback = setup_epsilon_callback
                
                # 开始训练
                self.training_thread.start()
                self.is_training = True
                
                # 启动探索率更新定时器
                self.epsilon_timer.start()
                print("[定时器] 已启动探索率更新定时器")

                # 更新UI状态
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.eval_best_btn.setEnabled(False)

                # 重置进度条
                self.progress_bar.setValue(0)
                self.episode_progress_bar.setValue(0)

                self.status_label.setText("训练中...")
                self.log_message(f"开始{model_type}模型训练，最大回合数: {max_episodes}")

            except Exception as e:
                import traceback
                error_message = traceback.format_exc()
                self.log_message(f"启动训练时出错: {str(e)}")
                print(f"ERROR: 启动训练时出错: {str(e)}\n{error_message}")
                QMessageBox.critical(self, "错误", f"无法启动训练: {str(e)}")
        
        except Exception as e:
            print(f"ERROR: 启动训练时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法启动训练: {str(e)}")

    def on_training_completed(self):
        """训练完成时的处理"""
        self.is_training = False

        # 停止探索率更新定时器
        if hasattr(self, 'epsilon_timer') and self.epsilon_timer.isActive():
            self.epsilon_timer.stop()
            print("[定时器] 已停止探索率更新定时器")

        # 重置停止请求标志
        if hasattr(self, 'stop_requested'):
            self.stop_requested = False

        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # 判断是否有找到最佳模型
        has_best_model = hasattr(self, 'best_model') and self.best_model is not None

        # 启用评估按钮，只有在找到最佳模型时
        self.eval_best_btn.setEnabled(has_best_model)

        # 更新状态标签
        if hasattr(self, 'stop_requested') and self.stop_requested:
            self.status_label.setText("训练已停止")
            self.log_message("训练已根据请求停止")
        else:
            self.status_label.setText("训练完成")
            self.log_message("训练完成")

        # 注意：不再在训练结束时重置历史数据和图表，让图表保持显示直到下次训练开始
        # self._reset_history_data()

        # 如果有最佳模型，更新显示
        if has_best_model:
            # 构建最佳模型信息消息
            model_info = f"最佳模型: 回合 {self.best_model_episode}"
            if hasattr(self, 'best_model_reward'):
                model_info += f", 奖励: {self.best_model_reward:.4f}"
            if hasattr(self, 'best_composite_score'):
                model_info += f", 综合得分: {self.best_composite_score:.4f}"

            # 记录日志
            self.log_message(model_info)

            # 显示提示信息
            QMessageBox.information(self, "训练完成",
                f"训练已完成，请点击'评估最佳模型'查看详细结果。\n{model_info}")
        else:
            self.log_message("未找到有效的最佳模型")
            QMessageBox.warning(self, "训练完成", "训练已完成，但未找到有效的最佳模型。")

    def on_chart_visibility_changed(self):
        """处理图表可见性变化事件"""
        # 更新图表显示控制变量
        self.show_rewards_plot = self.show_rewards_check.isChecked()
        self.show_returns_plot = self.show_returns_check.isChecked()
        self.show_learning_rate_plot = self.show_learning_rate_check.isChecked()
        self.show_portfolio_plot = self.show_portfolio_check.isChecked()

        # 记录日志
        self.log_message(f"图表显示设置已更新: 奖励曲线={self.show_rewards_plot}, 收益曲线={self.show_returns_plot}, 学习率曲线={self.show_learning_rate_plot}, 资产曲线={self.show_portfolio_plot}")

    def update_rewards_plot(self, rewards):
        """
        更新奖励曲线图

        参数:
            rewards: 奖励列表
        """
        # 如果图表被设置为不显示，则跳过更新
        if not self.show_rewards_plot:
            return

        if not self.matplotlib_available or not hasattr(self, 'rewards_ax'):
            return

        # 保存当前轴的范围
        current_xlim = self.rewards_ax.get_xlim()
        current_ylim = self.rewards_ax.get_ylim()

        # 更新奖励曲线
        self.rewards_ax.clear()

        # 创建正确的x轴数据点 - 考虑实际采样频率（每10步一个点）
        # 第一个点是步数1，之后每10步一个点 (1, 11, 21, 31...)
        # 使用存储的步数信息或生成新的步数序列
        if hasattr(self, 'rewards_steps') and len(self.rewards_steps) == len(rewards):
            # 如果有存储的实际步数，使用它们
            steps = self.rewards_steps
            print(f"DEBUG: 使用存储的奖励曲线步数，范围: {min(steps)} - {max(steps)}")
        else:
            # 如果没有存储的步数，生成正确的步数序列
            steps = np.arange(1, len(rewards)*10, 10)  # 生成正确的步数序列
            if len(steps) < len(rewards):  # 确保长度匹配
                steps = np.append(steps, steps[-1] + 10 if len(steps) > 0 else 1)
            print(f"DEBUG: 生成奖励曲线步数，范围: {min(steps) if steps else 0} - {max(steps) if steps else 0}")

        # 使用实际步数作为x轴，确保x轴刻度与训练步数一致
        self.rewards_ax.plot(steps, rewards)
        self.rewards_ax.set_title('训练奖励曲线')
        self.rewards_ax.set_xlabel('训练步数')
        self.rewards_ax.set_ylabel('累积奖励')
        self.rewards_ax.grid(True)

        # 打印调试信息
        print(f"DEBUG-PLOT: 奖励曲线图更新 - 数据点数量={len(rewards)}, x轴步数范围=[{steps[0] if len(steps) > 0 else 0}, {steps[-1] if len(steps) > 0 else 0}]")

        # 如果之前有设置过范围且数据点数量足够，尝试保持相同的视图
        if len(rewards) > 3 and current_xlim[1] > current_xlim[0]:
            # 智能调整X轴范围，使用实际步数
            if len(steps) > 0:
                max_step = steps[-1]
                new_xlim = (0, max(current_xlim[1], max_step * 1.1))
                self.rewards_ax.set_xlim(new_xlim)

            # 只有当之前有明确设置Y轴范围时才保持它
            if current_ylim[1] > current_ylim[0] and current_ylim[1] != 1.0:
                data_min = min(rewards)
                data_max = max(rewards)
                # 确保数据范围在视图内，必要时扩展
                new_ylim = (
                    min(current_ylim[0], data_min * 1.1),
                    max(current_ylim[1], data_max * 1.1)
                )
                self.rewards_ax.set_ylim(new_ylim)

        # 更新图表
        self.rewards_figure.tight_layout()
        self.rewards_canvas.draw()

    def update_learning_rate_plot(self, learning_rates):
        """更新学习率曲线图"""
        # 如果图表被设置为不显示，则跳过更新
        if not self.show_learning_rate_plot:
            return

        try:
            # 检查是否有数据
            if not learning_rates or len(learning_rates) == 0:
                print("DEBUG: 学习率数据为空，跳过更新")
                return
                
            print(f"DEBUG: 正在更新学习率曲线，数据点数={len(learning_rates)}")
                
            # 获取步数数据
            steps = None
            if hasattr(self, 'learning_rate_steps') and len(self.learning_rate_steps) >= len(learning_rates):
                offset = len(self.learning_rate_steps) - len(learning_rates)
                steps = self.learning_rate_steps[offset:offset+len(learning_rates)]
                print(f"DEBUG: 使用存储的步数，范围: {min(steps) if steps else 0} - {max(steps) if steps else 0}")

            # 确保learning_rate_plot对象存在
            if not hasattr(self, 'learning_rate_plot') or self.learning_rate_plot is None:
                from rl_strategies.ui.learning_rate_plot import LearningRatePlot
                self.learning_rate_plot = LearningRatePlot()
                print("DEBUG: 创建了新的学习率曲线图对象")
                
            # 更新曲线数据
            self.learning_rate_plot.update_plot(learning_rates, steps)
            print(f"DEBUG: 学习率曲线更新完成，数据点数={len(learning_rates)}")

        except Exception as e:
            print(f"ERROR: 更新学习率曲线时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def update_returns_plot(self, returns):
        """
        更新收益率曲线图

        参数:
            returns: 收益率列表
        """
        # 如果图表被设置为不显示，则跳过更新
        if not self.show_returns_plot:
            return

        if not hasattr(self, 'returns_plot'):
            return

        # 获取步数数据
        steps = None
        if hasattr(self, 'returns_steps') and len(self.returns_steps) == len(returns):
            steps = self.returns_steps

        # 更新PyQtGraph图表
        self.returns_plot.update_plot(returns, steps)

        # 不再自动切换到收益曲线标签页，让用户自由选择

    def update_metrics_text(self, metrics):
        """
        更新性能指标文本

        参数:
            metrics: 性能指标字典
        """
        # 格式化指标文本
        text = ""
        for key, value in metrics.items():
            # 跳过一些复杂的指标，如交易历史
            if key in ['trade_history', 'trades']:
                continue

            if isinstance(value, (int, float)):
                # 数值添加适当的格式
                if 'rate' in key.lower() or 'ratio' in key.lower() or 'percent' in key.lower():
                    text += f"{key}: {value:.2f}%\n"
                elif isinstance(value, float):
                    text += f"{key}: {value:.4f}\n"
                else:
                    text += f"{key}: {value}\n"
            else:
                # 非数值直接显示
                text += f"{key}: {value}\n"

        # 更新指标文本框
        self.metrics_text.setText(text)

    def init_plots(self):
        """初始化图表"""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

            # 奖励曲线图
            self.rewards_figure = Figure(figsize=(5, 4), dpi=100)
            self.rewards_canvas = FigureCanvas(self.rewards_figure)
            self.rewards_ax = self.rewards_figure.add_subplot(111)  # 修改为单一图表
            self.rewards_ax.set_title('训练奖励曲线')
            self.rewards_ax.set_xlabel('回合')
            self.rewards_ax.set_ylabel('累积奖励')
            self.rewards_ax.grid(True)

            # 替换奖励曲线占位符
            rewards_layout = QVBoxLayout()
            rewards_layout.addWidget(self.rewards_canvas)

            # 检查rewards_plot_widget是否存在
            if hasattr(self, 'rewards_plot_widget'):
                # 清除现有布局
                if self.rewards_plot_widget.layout():
                    QWidget().setLayout(self.rewards_plot_widget.layout())
                # 设置新布局
                self.rewards_plot_widget.setLayout(rewards_layout)

            # 创建收益率曲线图 (PyQtGraph版本)
            from rl_strategies.ui.returns_plot import ReturnsPlot
            self.returns_plot = ReturnsPlot()

            # 替换收益率曲线占位符
            returns_layout = QVBoxLayout()
            returns_layout.addWidget(self.returns_plot)

            # 检查returns_plot_widget是否存在
            if hasattr(self, 'returns_plot_widget'):
                # 清除现有布局
                if self.returns_plot_widget.layout():
                    QWidget().setLayout(self.returns_plot_widget.layout())
                # 设置新布局
                self.returns_plot_widget.setLayout(returns_layout)

            # 最佳模型评估图
            self.best_model_figure = Figure(figsize=(5, 4), dpi=100)
            self.best_model_canvas = FigureCanvas(self.best_model_figure)
            self.best_model_ax = self.best_model_figure.add_subplot(111)
            self.best_model_ax.set_title('最佳模型投资组合价值曲线')
            self.best_model_ax.set_xlabel('交易步数')
            self.best_model_ax.set_ylabel('资金金额')
            self.best_model_ax.grid(True)

            # 替换最佳模型图占位符
            if hasattr(self, 'best_model_plot_layout'):
                # 清除现有布局中的所有小部件
                while self.best_model_plot_layout.count():
                    item = self.best_model_plot_layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                # 添加画布到布局
                self.best_model_plot_layout.addWidget(self.best_model_canvas)

            self.log_message("图表初始化完成")

        except Exception as e:
            self.log_message(f"初始化图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_portfolio_curve(self, portfolio_values):
        """
        绘制资产曲线

        参数:
            portfolio_values: 资产价值列表
        """
        # 如果图表被设置为不显示，则跳过更新
        if not self.show_portfolio_plot:
            return

        if not self.matplotlib_available or not hasattr(self, 'best_model_ax'):
            self.log_message("matplotlib不可用，无法绘制图表")
            return

        try:
            # 清除之前的图表
            self.best_model_ax.clear()

            # 绘制资产曲线
            self.best_model_ax.plot(portfolio_values, 'b-')
            self.best_model_ax.set_title('资产价值曲线')
            self.best_model_ax.set_xlabel('交易步数')
            self.best_model_ax.set_ylabel('资产价值')
            self.best_model_ax.grid(True)

            # 添加初始资金线
            initial_value = portfolio_values[0] if portfolio_values else 0
            self.best_model_ax.axhline(y=initial_value, color='r', linestyle='--', alpha=0.7, label='初始资金')

            # 更新图表
            self.best_model_figure.tight_layout()
            self.best_model_canvas.draw()

        except Exception as e:
            self.log_message(f"绘制资产曲线时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_ui(self):
        """定时更新UI状态"""
        # 检查训练线程状态
        if self.is_training and self.training_thread:
            # 如果线程已完成但状态未更新
            if not self.training_thread.isRunning() and self.is_training:
                self.is_training = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.eval_best_btn.setEnabled(self.best_model is not None)
                self.status_label.setText("训练已停止")
                self.log_message("训练线程已停止")

    def set_kline_data(self, data):
        """
        设置K线数据

        参数:
            data: DataFrame类型的K线数据
        """
        if data is None or len(data) == 0:
            self.log_message("警告: 收到空的K线数据")
            return

        self.kline_data = data
        self.log_message(f"加载了{len(data)}条K线数据，时间范围: {data.index[0]} 到 {data.index[-1]}")

        # 启用相关控件
        self.start_btn.setEnabled(True)

        # 如果是首次加载数据，可以自动设置一些参数
        # 例如，可以根据数据长度调整训练/验证比例
        if len(data) > 10000:
            # 数据量大，可以使用更多的验证数据
            self.train_ratio_spin.setValue(80)
        elif len(data) < 1000:
            # 数据量小，增加训练数据比例
            self.train_ratio_spin.setValue(90)
        else:
            # 默认比例
            self.train_ratio_spin.setValue(70)

        # 记录数据加载时间，以便在训练时检查数据是否已更新
        self.data_loaded_time = datetime.now()

    def init_training_control_tab(self):
        """初始化训练控制标签页"""
        training_control_layout = QVBoxLayout()
        training_controls_group = QGroupBox("训练控制")

        # 创建训练参数布局
        form_layout = QFormLayout()

        # 最大回合数
        self.max_episodes_spin = QSpinBox()
        self.max_episodes_spin.setRange(1, 10000)
        self.max_episodes_spin.setValue(500)
        self.max_episodes_spin.setToolTip("设置最大训练回合数")
        form_layout.addRow("最大回合数:", self.max_episodes_spin)

        # 训练/验证比例
        self.train_val_ratio_spin = QDoubleSpinBox()
        self.train_val_ratio_spin.setRange(0.5, 0.95)
        self.train_val_ratio_spin.setValue(0.8)
        self.train_val_ratio_spin.setSingleStep(0.05)
        self.train_val_ratio_spin.setToolTip("设置训练数据占总数据的比例")
        form_layout.addRow("训练/总数据比例:", self.train_val_ratio_spin)

        # 添加到布局
        controls_layout = QVBoxLayout()
        controls_layout.addLayout(form_layout)

        # 创建训练控制按钮
        buttons_layout = QHBoxLayout()

        # 开始训练按钮
        self.start_button = QPushButton("开始训练")
        self.start_button.clicked.connect(self.start_training)
        buttons_layout.addWidget(self.start_button)

        # 停止训练按钮
        self.stop_button = QPushButton("停止训练")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)  # 初始禁用
        buttons_layout.addWidget(self.stop_button)

        # 评估最佳模型按钮
        self.eval_button = QPushButton("评估最佳模型")
        self.eval_button.clicked.connect(self.evaluate_best_model)
        self.eval_button.setEnabled(False)  # 初始禁用
        buttons_layout.addWidget(self.eval_button)

        # 添加到控件布局
        controls_layout.addLayout(buttons_layout)
        training_controls_group.setLayout(controls_layout)
        training_control_layout.addWidget(training_controls_group)

        # 训练状态组
        training_status_group = QGroupBox("训练状态")
        training_status_layout = QVBoxLayout()

        # 添加进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("总进度:"))
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        training_status_layout.addLayout(progress_layout)

        # 添加回合进度条
        episode_progress_layout = QHBoxLayout()
        episode_progress_layout.addWidget(QLabel("回合进度:"))
        self.episode_progress_bar = QProgressBar()
        episode_progress_layout.addWidget(self.episode_progress_bar)
        training_status_layout.addLayout(episode_progress_layout)

        # 添加状态标签
        status_layout = QGridLayout()
        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label, 0, 0)

        self.episode_label = QLabel("回合: 0/0")
        status_layout.addWidget(self.episode_label, 0, 1)

        self.step_label = QLabel("当前步骤: 0/0")
        status_layout.addWidget(self.step_label, 1, 0)

        self.reward_label = QLabel("当前奖励: 0.0000")
        status_layout.addWidget(self.reward_label, 1, 1)

        self.return_label = QLabel("收益率: 0.00%")
        status_layout.addWidget(self.return_label, 2, 0)

        # 添加学习率标签
        self.learning_rate_label = QLabel("学习率: 0.001000")
        self.learning_rate_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.learning_rate_label, 2, 1)

        # 添加最佳模型信息
        self.best_model_info = QLabel("最佳模型: 未找到")
        self.best_model_info.setStyleSheet("font-weight: bold; color: blue;")
        status_layout.addWidget(self.best_model_info, 3, 0, 1, 2)

        training_status_layout.addLayout(status_layout)
        training_status_group.setLayout(training_status_layout)
        training_control_layout.addWidget(training_status_group)

        # 设置标签页布局
        training_control_tab = QWidget()
        training_control_tab.setLayout(training_control_layout)

        return training_control_tab

    def init_model_config_tab(self):
        """初始化模型配置标签页"""
        model_config_layout = QVBoxLayout()

        # 模型选择组
        model_select_group = QGroupBox("模型选择")
        model_select_layout = QVBoxLayout()

        # 模型类型选择
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("模型类型:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["DQN", "PPO", "A2C"])
        self.model_type_combo.currentIndexChanged.connect(self.on_model_changed)
        model_type_layout.addWidget(self.model_type_combo)
        model_select_layout.addLayout(model_type_layout)

        # 模型参数组
        model_param_group = QGroupBox("模型参数")
        model_param_layout = QVBoxLayout()

        # 学习率
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学习率:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        lr_layout.addWidget(self.lr_spin)
        model_param_layout.addLayout(lr_layout)

        # 折扣因子
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("折扣因子(γ):"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.8, 0.999)
        self.gamma_spin.setSingleStep(0.001)
        self.gamma_spin.setDecimals(3)
        self.gamma_spin.setValue(0.99)
        gamma_layout.addWidget(self.gamma_spin)
        model_param_layout.addLayout(gamma_layout)

        # 批量大小
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("批量大小:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 512)
        self.batch_size_spin.setSingleStep(16)
        self.batch_size_spin.setValue(64)
        batch_layout.addWidget(self.batch_size_spin)
        model_param_layout.addLayout(batch_layout)

        # 添加缓冲区大小控件
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("回放缓冲区大小:"))
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(1000, 100000)
        self.buffer_size_spin.setSingleStep(1000)
        self.buffer_size_spin.setValue(10000)
        self.buffer_size_spin.setToolTip("设置经验回放缓冲区的大小，范围：1000-100000")
        buffer_layout.addWidget(self.buffer_size_spin)
        model_param_layout.addLayout(buffer_layout)

        # 训练模式
        train_mode_layout = QHBoxLayout()
        train_mode_layout.addWidget(QLabel("训练模式:"))
        self.train_mode_combo = QComboBox()
        self.train_mode_combo.addItems(["Online", "Batch", "Memory_Efficient"])
        self.train_mode_combo.setCurrentText("Online")
        self.train_mode_combo.setToolTip("选择训练模式:\nOnline: 在线学习\nBatch: 批量学习\nMemory_Efficient: 内存优化模式")
        train_mode_layout.addWidget(self.train_mode_combo)
        model_param_layout.addLayout(train_mode_layout)

        # 隐藏层配置
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("隐藏层:"))
        self.hidden_layers_text = QComboBox()
        self.hidden_layers_text.addItems([
            "64,32",
            "128,64",
            "256,128",
            "128,64,32",
            "256,128,64",
            "512,256,128"
        ])
        self.hidden_layers_text.setCurrentText("128,64")  # 设置默认值
        self.hidden_layers_text.setEditable(True)  # 允许用户输入自定义值
        self.hidden_layers_text.setToolTip("选择或输入隐藏层配置，用逗号分隔神经元数量")
        hidden_layout.addWidget(self.hidden_layers_text)
        model_param_layout.addLayout(hidden_layout)

        # 添加动态学习率适应选项
        self.dynamic_lr_check = QCheckBox("启用动态学习率")
        self.dynamic_lr_check.setChecked(True)
        self.dynamic_lr_check.setToolTip("根据收益情况自动调整学习率")
        model_param_layout.addWidget(self.dynamic_lr_check)

        # 学习率范围
        lr_range_layout = QHBoxLayout()
        lr_range_layout.addWidget(QLabel("最小学习率:"))
        self.min_lr_spin = QDoubleSpinBox()
        self.min_lr_spin.setRange(0.000001, 0.01)
        self.min_lr_spin.setSingleStep(0.00001)
        self.min_lr_spin.setDecimals(6)
        self.min_lr_spin.setValue(0.0001)  # 默认为基础学习率的1/10
        self.min_lr_spin.setToolTip("学习率的最小值")
        lr_range_layout.addWidget(self.min_lr_spin)

        lr_range_layout.addWidget(QLabel("最大学习率:"))
        self.max_lr_spin = QDoubleSpinBox()
        self.max_lr_spin.setRange(0.0001, 0.5)
        self.max_lr_spin.setSingleStep(0.001)
        self.max_lr_spin.setDecimals(6)
        self.max_lr_spin.setValue(0.005)  # 默认为基础学习率的5倍
        self.max_lr_spin.setToolTip("学习率的最大值")
        lr_range_layout.addWidget(self.max_lr_spin)
        model_param_layout.addLayout(lr_range_layout)

        # 自动更新学习率范围
        self.lr_spin.valueChanged.connect(self.update_lr_range)

        # 学习率调整阈值
        lr_threshold_layout = QHBoxLayout()
        lr_threshold_layout.addWidget(QLabel("增加阈值:"))
        self.lr_increase_threshold_spin = QDoubleSpinBox()
        self.lr_increase_threshold_spin.setRange(-1.0, 0.0)
        self.lr_increase_threshold_spin.setValue(-0.1)
        self.lr_increase_threshold_spin.setSingleStep(0.05)
        self.lr_increase_threshold_spin.setToolTip("收益率低于此值时提高学习率")
        lr_threshold_layout.addWidget(self.lr_increase_threshold_spin)

        lr_threshold_layout.addWidget(QLabel("减少阈值:"))
        self.lr_decrease_threshold_spin = QDoubleSpinBox()
        self.lr_decrease_threshold_spin.setRange(0.0, 1.0)
        self.lr_decrease_threshold_spin.setValue(0.05)
        self.lr_decrease_threshold_spin.setSingleStep(0.05)
        self.lr_decrease_threshold_spin.setToolTip("收益率高于此值时降低学习率")
        lr_threshold_layout.addWidget(self.lr_decrease_threshold_spin)
        model_param_layout.addLayout(lr_threshold_layout)

        # 学习率调整倍率
        lr_factor_layout = QHBoxLayout()
        lr_factor_layout.addWidget(QLabel("增加倍率:"))
        self.lr_increase_factor_spin = QDoubleSpinBox()
        self.lr_increase_factor_spin.setRange(1.1, 5.0)
        self.lr_increase_factor_spin.setValue(1.5)
        self.lr_increase_factor_spin.setSingleStep(0.1)
        self.lr_increase_factor_spin.setToolTip("学习率增加的倍数")
        lr_factor_layout.addWidget(self.lr_increase_factor_spin)

        lr_factor_layout.addWidget(QLabel("减少倍率:"))
        self.lr_decrease_factor_spin = QDoubleSpinBox()
        self.lr_decrease_factor_spin.setRange(0.1, 0.9)
        self.lr_decrease_factor_spin.setValue(0.8)
        self.lr_decrease_factor_spin.setSingleStep(0.05)
        self.lr_decrease_factor_spin.setToolTip("学习率减少的倍数")
        lr_factor_layout.addWidget(self.lr_decrease_factor_spin)
        model_param_layout.addLayout(lr_factor_layout)

        model_param_group.setLayout(model_param_layout)
        model_select_layout.addWidget(model_param_group)
        model_select_group.setLayout(model_select_layout)
        model_config_layout.addWidget(model_select_group)

        # 模型配置标签页
        model_config_tab = QWidget()
        model_config_tab.setLayout(model_config_layout)

        return model_config_tab

    def update_lr_range(self):
        """根据基础学习率更新学习率范围"""
        base_lr = self.lr_spin.value()

        # 如果这些控件已经初始化，则更新它们
        if hasattr(self, 'min_lr_spin') and hasattr(self, 'max_lr_spin'):
            # 将最小学习率设为基础学习率的1/10
            self.min_lr_spin.setValue(base_lr / 10.0)
            # 将最大学习率设为基础学习率的5倍
            self.max_lr_spin.setValue(base_lr * 5.0)

    def get_agent_config(self):
        """获取代理配置"""
        # 获取最大学习率值，优先使用模型选择标签页的设置
        max_lr = self.model_select_max_lr_spin.value() if hasattr(self, 'model_select_max_lr_spin') else self.max_lr_spin.value()

        # 打印调试信息
        print("\n=== 获取代理配置时的学习率设置 ===")
        print(f"[UI设置] 基础学习率: {self.lr_spin.value():.6f}")
        print(f"[UI设置] 模型选择标签页最大学习率: {self.model_select_max_lr_spin.value() if hasattr(self, 'model_select_max_lr_spin') else '不存在'}")
        print(f"[UI设置] 高级设置最大学习率: {self.max_lr_spin.value() if hasattr(self, 'max_lr_spin') else '不存在'}")
        print(f"[UI设置] 最终选择的最大学习率: {max_lr}")

        # 获取隐藏层配置
        try:
            if isinstance(self.hidden_layers_text, QComboBox):
                hidden_layers_str = self.hidden_layers_text.currentText()
            else:
                hidden_layers_str = self.hidden_layers_text.text()
            hidden_layers = [int(layer) for layer in hidden_layers_str.split(',')]
        except (AttributeError, ValueError) as e:
            print(f"警告: 获取隐藏层配置出错，使用默认值 [128,64]: {str(e)}")
            hidden_layers = [128, 64]

        # 获取训练模式
        try:
            train_mode = self.train_mode_combo.currentText().lower() if hasattr(self, 'train_mode_combo') else 'online'
        except AttributeError:
            print("警告: 无法获取训练模式，使用默认值 'online'")
            train_mode = 'online'

        # 基础配置
        config = {
            'learning_rate': self.lr_spin.value(),
            'gamma': self.gamma_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'buffer_size': getattr(self, 'buffer_size_spin', None).value() if hasattr(self, 'buffer_size_spin') else 10000,
            'hidden_layers': hidden_layers,
            'train_mode': train_mode,
            'max_learning_rate': max_lr,  # 设置最大学习率
            'lr_adaptation': {  # 将学习率自适应配置移到顶层
                'enabled': hasattr(self, 'dynamic_lr_check') and self.dynamic_lr_check.isChecked(),
                'base_lr': self.lr_spin.value(),
                'min_lr': self.min_lr_spin.value() if hasattr(self, 'min_lr_spin') else self.lr_spin.value() / 10.0,
                'max_lr': max_lr,  # 确保与顶层配置一致
                'increase_threshold': self.lr_increase_threshold_spin.value() if hasattr(self, 'lr_increase_threshold_spin') else -0.1,
                'decrease_threshold': self.lr_decrease_threshold_spin.value() if hasattr(self, 'lr_decrease_threshold_spin') else 0.05,
                'increase_factor': self.lr_increase_factor_spin.value() if hasattr(self, 'lr_increase_factor_spin') else 1.5,
                'decrease_factor': self.lr_decrease_factor_spin.value() if hasattr(self, 'lr_decrease_factor_spin') else 0.8,
                'adaptation_window': 5,
                'cooldown_period': 3
            }
        }

        # 打印详细的配置信息
        print("\n=== 最终生成的代理配置 ===")
        print(f"[配置] 基础学习率: {config['learning_rate']:.6f}")
        print(f"[配置] 最大学习率(max_learning_rate): {config['max_learning_rate']:.6f}")
        print(f"[配置] 隐藏层配置: {config['hidden_layers']}")
        print(f"[配置] 训练模式: {config['train_mode']}")
        print(f"[配置] 学习率自适应配置:")
        print(f"  - 启用状态: {config['lr_adaptation']['enabled']}")
        print(f"  - 基础学习率: {config['lr_adaptation']['base_lr']:.6f}")
        print(f"  - 最小学习率: {config['lr_adaptation']['min_lr']:.6f}")
        print(f"  - 最大学习率: {config['lr_adaptation']['max_lr']:.6f}")
        print("="*50)

        return config

    def compress_data(self, data, steps):
        """
        压缩数据，保留关键点
        
        参数:
            data: 需要压缩的数据
            steps: 对应的步数
            
        返回:
            (compressed_data, compressed_steps): 压缩后的数据和步数
        """
        if len(data) <= 100:
            return data, steps
            
        result_data = []
        result_steps = []
        
        # 保留第一个点
        result_data.append(data[0])
        result_steps.append(steps[0])
        
        # 计算压缩区间
        interval_size = len(data) // 100
        
        # 压缩中间点，每个区间计算平均值
        for i in range(1, 99):
            start_idx = i * interval_size
            end_idx = min((i + 1) * interval_size, len(data) - 1)
            
            # 确保区间有效
            if start_idx >= end_idx:
                continue
                
            # 计算区间平均值
            interval_data = data[start_idx:end_idx]
            result_data.append(sum(interval_data) / len(interval_data))
            
            # 取区间中间的步数
            middle_idx = (start_idx + end_idx) // 2
            result_steps.append(steps[middle_idx])
            
        # 保留最后一个点
        if len(data) > 1 and (not result_data or result_data[-1] != data[-1]):
            result_data.append(data[-1])
            result_steps.append(steps[-1])
            
        print(f"数据压缩: {len(data)}点 -> {len(result_data)}点")
        return result_data, result_steps

    def update_epsilon_from_agent(self):
        """从代理对象获取当前探索率并更新UI"""
        try:
            # 检查是否正在训练
            if not self.is_training or not hasattr(self, 'training_thread') or not self.training_thread.isRunning():
                return
                
            # 获取当前探索率和步数
            step_counter = 0
            current_epsilon = 0.0
            
            # 首先使用UI侧的步数计数器
            if hasattr(self, 'ui_step_counter'):
                step_counter = self.ui_step_counter
                print(f"使用UI侧步数计数器: {step_counter}")
            
            # 首先使用UI侧维护的探索率
            if hasattr(self, 'ui_epsilon_value'):
                current_epsilon = self.ui_epsilon_value
                print(f"使用UI侧探索率值: {current_epsilon}")
            
            # 尝试从训练器对象获取步数（如果UI计数器为0）
            if step_counter == 0 and hasattr(self.training_thread, 'trainer') and self.training_thread.trainer is not None:
                trainer = self.training_thread.trainer
                
                # 尝试不同的步数属性名
                if hasattr(trainer, 'current_step'):
                    step_counter = trainer.current_step
                    print(f"从trainer.current_step获取步数: {step_counter}")
                elif hasattr(trainer, 'step_count'):
                    step_counter = trainer.step_count
                    print(f"从trainer.step_count获取步数: {step_counter}")
                elif hasattr(trainer, 'total_steps'):
                    step_counter = trainer.total_steps
                    print(f"从trainer.total_steps获取步数: {step_counter}")
            
            # 如果无法从训练器获取步数，再尝试从代理获取
            if (step_counter == 0 or current_epsilon == 0.0) and hasattr(self.training_thread, 'agent') and self.training_thread.agent is not None:
                agent = self.training_thread.agent
                print(f"尝试从代理获取信息...")
                
                # 检查代理是否有epsilon属性，如果UI侧探索率为0，则尝试从代理获取
                if hasattr(agent, 'epsilon') and (current_epsilon == 0.0 or abs(current_epsilon) < 1e-6):
                    agent_epsilon = agent.epsilon
                    print(f"从agent.epsilon获取探索率: {agent_epsilon}")
                    
                    # 只有当代理探索率不为0时才更新UI探索率
                    if agent_epsilon > 0.0:
                        current_epsilon = agent_epsilon
                        # 更新UI变量
                        self.ui_epsilon_value = current_epsilon
                        print(f"更新UI探索率为代理值: {self.ui_epsilon_value}")
                
                # 尝试获取步数
                if hasattr(agent, 'learn_step_counter'):
                    agent_step = agent.learn_step_counter
                    print(f"从agent.learn_step_counter获取步数: {agent_step}")
                    # 只有当代理步数大于当前步数时才更新
                    if agent_step > step_counter:
                        step_counter = agent_step
                        self.ui_step_counter = step_counter
                        print(f"更新UI步数为代理值: {self.ui_step_counter}")
            
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 创建探索率信息消息
            epsilon_message = f"【{current_time}】【步数: {step_counter}】探索率：{current_epsilon:.4f}，定时获取"
            print(f"[定时器] {epsilon_message}")
            
            # 更新UI
            if hasattr(self, 'epsilon_text') and self.epsilon_text is not None:
                # 添加消息到文本框
                self.epsilon_text.append(epsilon_message)
                # 滚动到底部
                self.epsilon_text.verticalScrollBar().setValue(self.epsilon_text.verticalScrollBar().maximum())
                
                # 更新探索率标签
                if hasattr(self, 'epsilon_info'):
                    self.epsilon_info.setText(f"当前探索率: {current_epsilon:.6f}")
                    
            # 强制刷新UI
            QApplication.processEvents()
            
        except Exception as e:
            print(f"[定时器] 更新探索率信息时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def stop_training(self):
        """停止训练过程"""
        # 检查是否已经在训练状态
        if not self.is_training:
            return

        # 检查是否已经发送了停止请求，避免重复发送
        if hasattr(self, 'stop_requested') and self.stop_requested:
            self.log_message("已经发送停止请求，请耐心等待当前轮次结束...")
            return

        # 设置停止状态标志
        self.stop_requested = True

        # 更新UI状态 - 禁用停止按钮防止重复点击
        self.stop_btn.setEnabled(False)
        self.status_label.setText("正在停止训练（等待当前轮次结束）...")
        self.log_message("正在停止训练，等待当前训练轮次结束...")

        # 请求停止训练 - 先尝试训练线程的stop方法
        if hasattr(self.training_thread, 'stop'):
            self.training_thread.stop()
            print("DEBUG: 通过训练线程发送停止信号")

        # 如果有直接访问的trainer对象，也设置它的停止标志
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.trainer.stop_requested = True
            print("DEBUG: 直接设置trainer的stop_requested=True")

        # 停止探索率更新定时器
        if hasattr(self, 'epsilon_timer') and self.epsilon_timer.isActive():
            self.epsilon_timer.stop()
            print("[定时器] 在停止训练时停止探索率更新定时器")

        # 移除重置历史数据的定时调用
        # QTimer.singleShot(1000, self._reset_history_data)  # 1秒后重置历史数据

    def update_epsilon_info(self, message):
        """更新探索率信息"""
        # 打印更详细的调试信息
        print(f"【探索率回调】收到消息: {message}")
        
        try:
            # 确保文本区域存在并可见
            if hasattr(self, 'epsilon_text') and self.epsilon_text is not None:
                print(f"【探索率回调】文本区域检查通过，准备更新UI")
                # 将探索率变化信息添加到文本框
                self.epsilon_text.append(message)
                print(f"【探索率回调】已添加消息到文本框")
                # 滚动到底部
                self.epsilon_text.verticalScrollBar().setValue(self.epsilon_text.verticalScrollBar().maximum())
                
                # 更新探索率信息标签
                if "探索率：" in message:
                    try:
                        # 提取探索率值
                        epsilon_value = float(message.split("探索率：")[1].split("，")[0])
                        # 更新UI变量
                        self.ui_epsilon_value = epsilon_value
                        print(f"【探索率回调】已更新UI探索率值: {self.ui_epsilon_value:.6f}")
                        
                        # 更新标签
                        self.epsilon_info.setText(f"当前探索率: {epsilon_value:.6f}")
                        print(f"【探索率回调】已更新探索率信息标签: {epsilon_value:.6f}")
                    except Exception as e:
                        print(f"【探索率回调】解析探索率信息时出错: {str(e)}")
            else:
                print("【探索率回调】错误: epsilon_text不存在或为None")
                
            # 强制更新UI
            QApplication.processEvents()
            print(f"【探索率回调】UI已刷新")
        except Exception as e:
            print(f"【探索率回调】更新探索率信息时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())