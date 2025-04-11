"""
可视化工具

用于可视化强化学习训练结果和交易绩效
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime


class Visualizer:
    """可视化工具类"""
    
    @staticmethod
    def plot_training_history(
        history: List[Dict], 
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制训练历史曲线
        
        参数:
            history: 训练历史记录
            metrics: 要绘制的指标列表
            figsize: 图像大小
            save_path: 保存路径
        """
        if not history:
            print("没有训练历史记录可供绘制")
            return
        
        # 如果未指定指标，则使用所有可用指标
        if metrics is None:
            # 排除时间和步数等非指标字段
            exclude_keys = ['episode', 'time', 'steps']
            metrics = [key for key in history[0].keys() if key not in exclude_keys]
        
        # 创建图像
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        # 提取episode数据
        episodes = [entry.get('episode', i+1) for i, entry in enumerate(history)]
        
        # 为每个指标绘制曲线
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 提取指标数据
            values = [entry.get(metric, 0) for entry in history]
            
            # 绘制曲线
            ax.plot(episodes, values, 'b-', linewidth=1.5)
            
            # 添加滑动平均曲线
            window_size = min(len(values) // 5, 20)
            if window_size > 1:
                values_series = pd.Series(values)
                smoothed = values_series.rolling(window=window_size).mean()
                ax.plot(episodes, smoothed, 'r-', linewidth=2)
            
            # 设置标题和标签
            ax.set_title(f'{metric.replace("_", " ").title()} over Training')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴标签
        axes[-1].set_xlabel('Episode')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 显示图像
        plt.show()
    
    @staticmethod
    def plot_learning_curve(
        eval_history: List[Dict],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制学习曲线
        
        参数:
            eval_history: 评估历史记录
            figsize: 图像大小
            save_path: 保存路径
        """
        if not eval_history:
            print("没有评估历史记录可供绘制")
            return
        
        # 创建图像
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 提取数据
        episodes = [entry.get('episode', i+1) for i, entry in enumerate(eval_history)]
        returns = [entry.get('avg_return', 0) for entry in eval_history]
        rewards = [entry.get('avg_reward', 0) for entry in eval_history]
        
        # 绘制收益率曲线
        color = 'tab:blue'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return (%)', color=color)
        ax1.plot(episodes, returns, 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 创建第二个y轴
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Reward', color=color)
        ax2.plot(episodes, rewards, 's-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 设置标题
        plt.title('Learning Curve: Return and Reward over Episodes')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, ['Return', 'Reward'], loc='best')
        
        # 添加网格
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 显示图像
        plt.show()
    
    @staticmethod
    def plot_portfolio_performance(
        history: List[Dict],
        benchmark: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制投资组合表现
        
        参数:
            history: 交易历史记录
            benchmark: 基准收益率（可选）
            figsize: 图像大小
            save_path: 保存路径
        """
        if not history:
            print("没有交易历史记录可供绘制")
            return
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # 提取数据
        steps = [entry.get('step', i) for i, entry in enumerate(history)]
        portfolio_values = [entry.get('portfolio_value', 0) for entry in history]
        actions = [entry.get('action', 'HOLD') for entry in history]
        
        # 计算收益率
        initial_value = portfolio_values[0]
        returns = [(value / initial_value - 1) * 100 for value in portfolio_values]
        
        # 绘制投资组合价值曲线
        ax1.plot(steps, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
        
        # 如果有基准收益率，也绘制它
        if benchmark is not None:
            # 确保基准长度与历史记录匹配
            if len(benchmark) >= len(steps):
                benchmark_values = benchmark.iloc[steps].values
                benchmark_values = benchmark_values * (initial_value / benchmark_values[0])
                ax1.plot(steps, benchmark_values, 'r--', linewidth=1.5, label='Benchmark')
        
        # 绘制交易点
        buy_steps = [steps[i] for i, action in enumerate(actions) if action == 'BUY']
        buy_values = [portfolio_values[i] for i, action in enumerate(actions) if action == 'BUY']
        sell_steps = [steps[i] for i, action in enumerate(actions) if action == 'SELL']
        sell_values = [portfolio_values[i] for i, action in enumerate(actions) if action == 'SELL']
        
        ax1.plot(buy_steps, buy_values, '^', markersize=8, color='g', label='Buy')
        ax1.plot(sell_steps, sell_values, 'v', markersize=8, color='r', label='Sell')
        
        # 设置标题和标签
        ax1.set_title('Portfolio Performance')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 绘制收益率变化
        ax2.plot(steps, returns, 'g-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 设置标签
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 显示图像
        plt.show()
    
    @staticmethod
    def plot_trade_distribution(
        history: List[Dict],
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制交易分布
        
        参数:
            history: 交易历史记录
            figsize: 图像大小
            save_path: 保存路径
        """
        if not history:
            print("没有交易历史记录可供绘制")
            return
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 提取动作数据
        actions = [entry.get('action', 'HOLD') for entry in history]
        
        # 计算每种动作的数量
        action_counts = {}
        for action in actions:
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
        
        # 绘制动作分布饼图
        labels = list(action_counts.keys())
        sizes = list(action_counts.values())
        colors = ['red', 'blue', 'green']
        explode = [0.1 if label != 'HOLD' else 0 for label in labels]
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title('Action Distribution')
        
        # 提取交易收益数据
        profits = []
        prev_portfolio_value = None
        
        for entry in history:
            if entry.get('action') != 'HOLD':
                if prev_portfolio_value is not None:
                    portfolio_value = entry.get('portfolio_value', 0)
                    profit = (portfolio_value - prev_portfolio_value) / prev_portfolio_value * 100
                    profits.append(profit)
                prev_portfolio_value = entry.get('portfolio_value', 0)
        
        # 绘制交易收益直方图
        if profits:
            ax2.hist(profits, bins=20, color='blue', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', linewidth=1)
            ax2.set_title('Trade Profit Distribution')
            ax2.set_xlabel('Profit (%)')
            ax2.set_ylabel('Number of Trades')
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'No trade data available', ha='center', va='center')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 显示图像
        plt.show()
    
    @staticmethod
    def save_training_summary(
        training_history: List[Dict],
        eval_history: List[Dict],
        save_dir: str,
        model_name: str
    ):
        """
        保存训练摘要
        
        参数:
            training_history: 训练历史记录
            eval_history: 评估历史记录
            save_dir: 保存目录
            model_name: 模型名称
        """
        # 创建保存目录
        summary_dir = os.path.join(save_dir, model_name, 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        # 保存训练曲线
        Visualizer.plot_training_history(
            training_history, 
            metrics=['reward', 'portfolio_value'], 
            save_path=os.path.join(summary_dir, 'training_history.png')
        )
        
        # 保存学习曲线
        if eval_history:
            Visualizer.plot_learning_curve(
                eval_history,
                save_path=os.path.join(summary_dir, 'learning_curve.png')
            )
        
        # 生成文本摘要
        if training_history and eval_history:
            last_train = training_history[-1]
            best_eval = max(eval_history, key=lambda x: x.get('avg_return', 0))
            
            summary_text = f"""Training Summary for {model_name}
            
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Training Statistics:
- Total Episodes: {last_train.get('episode', 0)}
- Final Reward: {last_train.get('reward', 0):.4f}
- Final Portfolio Value: {last_train.get('portfolio_value', 0):.2f}
- Total Trade Count: {last_train.get('trade_count', 0)}

Best Evaluation Results:
- Best Return: {best_eval.get('avg_return', 0):.2f}%
- Win Rate: {best_eval.get('win_rate', 0):.2f}%
- Average Trade Count: {best_eval.get('avg_trade_count', 0):.1f}
- Episode: {best_eval.get('episode', 0)}
            """
            
            # 保存摘要文本
            with open(os.path.join(summary_dir, 'summary.txt'), 'w') as f:
                f.write(summary_text)
            
            print(f"训练摘要已保存到 {summary_dir}")
        else:
            print("没有足够的历史记录生成摘要") 