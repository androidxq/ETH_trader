"""
强化学习训练器模块

该模块提供了训练和评估强化学习代理的功能
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
import random

# 导入RL组件
from rl_strategies.environments.trading_env import TradingEnv
from rl_strategies.agents.dqn_agent import DQNAgent
from rl_strategies.agents.ppo_agent import PPOAgent
from rl_strategies.agents.a2c_agent import A2CAgent
from rl_strategies.config import MODEL_SAVE_PATH


class RLTrainer:
    """
    强化学习代理训练器

    负责训练和评估RL代理，并提供训练进度和结果的回调
    """

    def __init__(
        self,
        agent_type: str,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame = None,
        env_config: Dict = None,
        agent_config: Dict = None,
        model_name: str = None
    ):
        """
        初始化训练器

        参数:
            agent_type: 代理类型 ('dqn', 'ppo', 'a2c')
            train_data: 训练数据
            val_data: 验证数据 (可选)
            env_config: 环境配置
            agent_config: 代理配置
            model_name: 模型名称 (可选，用于保存/加载)
        """
        self.agent_type = agent_type.lower()
        self.train_data = train_data
        self.val_data = val_data if val_data is not None else train_data
        self.env_config = env_config or {}
        self.agent_config = agent_config or {}

        # 验证和记录奖励参数
        self._validate_and_log_reward_params()

        # 设置模型名称 (用于保存和加载)
        self.model_name = model_name or f"{self.agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建训练环境 - 使用新的参数结构
        # 创建训练环境前记录配置
        print("\n=== 训练环境配置 ===")
        print(f"奖励类型: {self.env_config.get('reward_type', 'default')}")
        if self.env_config.get('reward_type') == 'compound':
            print("奖励权重:")
            for key, value in self.env_config.get('reward_weights', {}).items():
                print(f"  - {key}: {value}")
        
        self.train_env = TradingEnv(
            df=self.train_data,
            **self.env_config  # 直接传递完整的环境配置
        )

        # 创建验证环境 - 使用新的参数结构
        self.val_env = TradingEnv(
            df=self.val_data,
            **self.env_config  # 直接传递完整的环境配置
        )

        # 如果使用复合奖励，设置奖励权重
        if 'reward_weights' in self.env_config and self.env_config.get('reward_type') == 'compound':
            self.train_env.reward_weights = self.env_config['reward_weights']
            self.val_env.reward_weights = self.env_config['reward_weights']

        # 创建代理
        self._create_agent()

        # 训练状态
        self.is_initialized = False  # 初始化标志，初始为False
        self.is_training = False
        self.stop_requested = False
        self.stop_immediately = False
        self.force_terminate = False
        self.episodes_completed = 0
        self.best_model_reward = -float('inf')
        self.training_history = []
        self.eval_history = []
        self.progress_callback = None
        self.best_model_weights = None
        self.trade_records = []  # 初始化交易记录列表

        # 确保保存路径存在
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        self.eval_frequency = self.agent_config.get('eval_frequency', 10)  # 每10个回合评估一次
        self.verbose = self.agent_config.get('verbose', True)

        # 设置最大步数
        self.max_steps = self.env_config.get('max_episode_steps', 5000)

        # 设置训练模式
        self.train_mode = self.agent_config.get('train_mode', 'balanced')

        # 设置强制探索比例
        self.force_explore_ratio = self.agent_config.get('force_explore_ratio', 0.5)

        # 用于跟踪多个最佳模型
        self.top_models = []  # 存储(得分, 权重, 回合数)的列表
        self.max_top_models = 5  # 保存前5个最佳模型

        print("训练器初始化完成，准备好进行训练")

    def initialize(self):
        """
        完成训练器的初始化
        此方法用于确保训练器在开始训练前已经正确初始化
        """
        if self.is_initialized:
            print("训练器已经初始化")
            return True

        try:
            # 进行需要的初始化操作
            # 记录开始时间
            self.start_time = time.time()

            # 记录训练总步数
            self.total_steps_completed = 0

            # 标记为已初始化
            self.is_initialized = True
            print("训练器初始化成功")
            return True
        except Exception as e:
            print(f"训练器初始化失败: {str(e)}")
            return False

    def _create_agent(self):
        """创建指定类型的代理"""
        if self.agent_type == 'dqn':
            # 确保学习率适应参数正确传递给代理
            if 'lr_adaptation' in self.agent_config:
                # 打印学习率适应参数，便于调试
                print(f"学习率适应参数: {self.agent_config['lr_adaptation']}")

            # 打印完整的代理配置
            print("\n=== 训练器创建DQN代理的配置 ===")
            print(f"基础学习率: {self.agent_config.get('learning_rate')}")
            print(f"最大学习率: {self.agent_config.get('max_learning_rate')}")
            print(f"学习率自适应配置: {self.agent_config.get('lr_adaptation')}")
            print("="*50)

            self.agent = DQNAgent(
                state_dim=self.train_env.observation_space.shape[0],
                action_dim=self.train_env.action_space.n,
                config=self.agent_config  # 直接传递完整配置
            )
        elif self.agent_type == 'ppo':
            self.agent = PPOAgent(
                state_dim=self.train_env.observation_space.shape[0],
                action_dim=self.train_env.action_space.n,
                config=self.agent_config
            )
        elif self.agent_type == 'a2c':
            self.agent = A2CAgent(
                state_dim=self.train_env.observation_space.shape[0],
                action_dim=self.train_env.action_space.n,
                config=self.agent_config
            )
        else:
            raise ValueError(f"不支持的代理类型: {self.agent_type}")

    def _validate_and_log_reward_params(self):
        """验证和记录奖励参数"""
        print("\n=== 验证奖励参数 ===")
        if 'reward_type' not in self.env_config:
            print("警告: 未设置reward_type，将使用默认值")
            return

        reward_type = self.env_config['reward_type']
        print(f"奖励类型: {reward_type}")

        if reward_type == 'compound':
            reward_weights = self.env_config.get('reward_weights', {})
            if not reward_weights:
                print("警告: 使用复合奖励但未设置权重")
            else:
                print("奖励权重配置:")
                for key, value in reward_weights.items():
                    print(f"  - {key}: {value}")

        # 验证其他奖励相关参数
        reward_scaling = self.env_config.get('reward_scaling')
        if reward_scaling is not None:
            print(f"奖励缩放: {reward_scaling}")

        if self.env_config.get('penalize_inaction'):
            print(f"惩罚不活动: {self.env_config.get('inaction_penalty')}")

    def train(self, episodes, eval_interval=10, verbose=True):
        """训练模型"""
        self.verbose = verbose  # 设置verbose属性
        self.is_training = True  # 标记训练已开始
        print(f"\n[训练] 开始训练，总回合数: {episodes}")

        for episode in range(episodes):
            # 在每个轮次开始前检查停止标志
            if self.stop_requested:
                print(f"[训练] 检测到停止请求，结束训练过程")
                print(f"[训练] 已完成 {self.episodes_completed}/{episodes} 轮次")
                break

            # 设置当前回合信息，便于跟踪
            self.current_episode = episode + 1

            print(f"\n[训练] 开始第 {episode + 1}/{episodes} 回合 (总完成: {self.episodes_completed})")

            # 训练一个回合
            episode_result = self.train_episode(episode)

            # 处理train_episode返回的结果（可能是元组或字典）
            result_dict = None
            if isinstance(episode_result, tuple) and len(episode_result) == 2:
                success, data = episode_result
                if not success:
                    print(f"[训练] 回合 {episode + 1} 训练失败: {data.get('error', '未知错误')}")
                    continue
                result_dict = data
            elif isinstance(episode_result, dict):
                # 直接返回的是字典
                result_dict = episode_result
            else:
                print(f"[训练] 警告: 无法处理回合 {episode + 1} 的训练结果，跳过该回合")
                continue

            # 回合完成后，再次检查停止标志
            if self.stop_requested:
                print(f"[训练] 回合 {episode + 1} 完成后检测到停止请求，终止训练")
                # 发送最终进度
                if self.progress_callback and result_dict:
                    result_dict['final'] = True  # 标记为最终更新
                    self.progress_callback(result_dict)
                break

            # 发送训练进度
            if self.progress_callback and result_dict:
                self.progress_callback(result_dict)

            # 定期评估
            if (episode + 1) % eval_interval == 0:
                try:
                    print(f"[训练] 开始第 {episode + 1} 回合的评估")
                    eval_results = self.evaluate(episodes=1)

                    # 更新模型得分和排名
                    if 'avg_reward' in eval_results and 'avg_return' in eval_results:
                        # 计算综合得分 (可以根据需要调整权重)
                        train_reward = result_dict.get('reward', 0)
                        eval_reward = eval_results.get('avg_reward', 0)
                        eval_return = eval_results.get('avg_return', 0)
                        win_rate = eval_results.get('win_rate', 0)

                        # 综合得分计算公式：训练奖励*0.3 + 评估奖励*0.3 + 评估收益率*0.3 + 胜率*0.1
                        composite_score = (
                            train_reward * 0.3 +
                            eval_reward * 0.3 +
                            eval_return * 0.3 +
                            win_rate * 0.1
                        )

                        # 获取当前模型权重
                        current_weights = self.agent.get_weights()

                        # 添加到top_models列表并排序
                        self.update_top_models(composite_score, current_weights, episode+1)

                        # 如果是最佳模型，更新标记
                        if composite_score > self.best_model_reward:
                            self.best_model_reward = composite_score
                            self.best_model_weights = current_weights
                            print(f"[训练] 找到新的最佳模型! 回合: {episode+1}, 综合得分: {composite_score:.4f}")

                            # 添加最佳模型信息到结果中，供UI显示
                            if self.progress_callback:
                                best_model_info = {
                                    'best_model': True,
                                    'best_reward': self.best_model_reward,
                                    'best_episode': episode+1,
                                    'best_composite_score': composite_score,
                                    'best_eval_return': eval_return,
                                    'best_win_rate': win_rate
                                }
                                self.progress_callback(best_model_info)

                    print(f"[训练] 评估结果 - 平均奖励: {eval_results['avg_reward']:.4f}, 平均收益率: {eval_results['avg_return']:.2f}%, 胜率: {eval_results['win_rate']:.2f}%")
                except Exception as e:
                    print(f"[训练] 评估出错: {str(e)}")
                    # 继续训练，不中断
                    continue

        # 标记训练已结束
        self.is_training = False

        # 训练结束，保存最佳模型
        if self.best_model_weights is not None:
            self.agent.set_weights(self.best_model_weights)
            self.save_model(f"{self.model_name}_best")

            if self.verbose:
                print(f"[训练] 训练结束，已保存最佳模型")

            # 保存top N个模型
            for i, (score, weights, ep) in enumerate(self.top_models[:min(3, len(self.top_models))]):
                self.agent.set_weights(weights)
                self.save_model(f"{self.model_name}_top{i+1}_ep{ep}")
                print(f"[训练] 已保存第{i+1}名模型，回合{ep}，得分{score:.4f}")
        else:
            print("[训练] 警告: 未找到有效的最佳模型，无法保存")

        # 保存最终模型
        self.save_model(self.model_name)

        if self.verbose:
            print(f"[训练] 训练结束，已保存最终模型")

        return self.training_history

    def evaluate(self, episodes=1, verbose=True) -> Dict:
        """
        评估当前策略性能

        参数:
            episodes: 评估的回合数
            verbose: 是否打印详细信息

        返回:
            包含评估结果的字典
        """
        if not self.is_initialized:
            self.initialize()

        # 重置验证环境
        self.val_env.reset()

        # 清空交易历史
        trade_history = []

        # 总回报和交易次数
        total_reward = 0
        total_returns = 0
        total_trades = 0
        wins = 0  # 盈利交易的次数

        try:
            # 执行评估
            for episode in range(episodes):
                # 兼容新版gymnasium接口，reset返回(state, info)元组
                reset_result = self.val_env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) >= 1:
                    state = reset_result[0]  # 提取状态
                else:
                    # 向后兼容旧接口，直接返回状态
                    state = reset_result

                done = False
                truncated = False
                episode_reward = 0
                step = 0

                # 初始资产
                initial_portfolio_value = self.val_env.balance + self.val_env.position_value

                # 记录交易历史
                if hasattr(self.val_env, 'transaction_history'):
                    self.val_env.transaction_history = []

                while not done and not truncated and step < self.max_steps:
                    # 获取动作（无探索）
                    action = self.agent.act(state, explore=False)

                    # 执行动作
                    next_state, reward, done, truncated, info = self.val_env.step(action)

                    # 累加奖励
                    episode_reward += reward
                    step += 1

                    # 更新状态
                    state = next_state

                # 计算回合收益
                final_portfolio_value = self.val_env.balance + self.val_env.position_value
                episode_return = (final_portfolio_value / initial_portfolio_value - 1) * 100

                # 累计总收益和交易次数
                total_reward += episode_reward
                total_returns += episode_return
                total_trades += self.val_env.trade_count

                # 判断是否盈利
                if final_portfolio_value > initial_portfolio_value:
                    wins += 1

                # 收集交易历史
                if hasattr(self.val_env, 'transaction_history'):
                    # 直接添加完整的交易记录，确保所有字段都被保留
                    trade_history.extend(self.val_env.transaction_history)

                    # 调试打印
                    if len(self.val_env.transaction_history) > 0:
                        print(f"评估环境交易记录示例: {self.val_env.transaction_history[0]}")
                        # 检查关键字段
                        keys = set()
                        for record in self.val_env.transaction_history[:3]:
                            keys.update(record.keys())
                        print(f"交易记录字段: {keys}")

                if verbose:
                    print(f"评估回合 {episode+1}/{episodes}: 奖励 = {episode_reward:.4f}, 收益率 = {episode_return:.2f}%, 交易次数 = {self.val_env.trade_count}")

        except Exception as e:
            print(f"评估过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}

        # 计算平均值
        avg_reward = total_reward / episodes if episodes > 0 else 0
        avg_return = total_returns / episodes if episodes > 0 else 0
        avg_trades = total_trades / episodes if episodes > 0 else 0
        win_rate = (wins / episodes) * 100 if episodes > 0 else 0

        # 添加到评估历史
        eval_result = {
            'episode': self.episodes_completed,
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'avg_trade_count': avg_trades,
            'win_rate': win_rate
        }
        self.eval_history.append(eval_result)

        # 构建完整结果
        results = {
            'episode': self.episodes_completed,
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'avg_trade_count': avg_trades,
            'win_rate': win_rate,
            'trade_history': trade_history,
            'trades': trade_history  # 为了兼容性，提供两个相同的字段
        }

        # 如果使用了复合奖励，添加奖励组成
        if hasattr(self.val_env, 'last_reward_components'):
            results['reward_components'] = self.val_env.last_reward_components

        return results

    def stop_training(self):
        """停止训练 - 在当前轮次结束后停止"""
        if not self.is_training:
            print("训练已经停止，无需重复操作")
            return

        if self.stop_requested:
            print("已经请求停止训练，请等待当前轮次结束")
            return

        print("请求停止训练，将在当前轮次结束后停止...")
        self.stop_requested = True

        # 打印当前训练状态，帮助调试
        if hasattr(self, 'episodes_completed'):
            print(f"当前已完成轮次: {self.episodes_completed}")

        # 尝试保存当前模型状态（如果已训练至少一个轮次）
        if getattr(self, 'episodes_completed', 0) > 0:
            try:
                self.save_model(f"{self.model_name}_interrupted")
                print(f"已将当前训练状态保存到: {self.model_name}_interrupted")
            except Exception as e:
                print(f"警告: 保存中断状态失败: {str(e)}")

        return True

    def save_model(self, filename=None):
        """
        保存模型和训练历史

        参数:
            filename: 保存文件名（不包含路径和扩展名）
        """
        if filename is None:
            filename = self.model_name

        try:
            # 创建完整的保存路径
            save_path = os.path.join(MODEL_SAVE_PATH, filename)

            # 检查并创建目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 保存模型权重
            self.agent.save(f"{save_path}_weights.h5")

            # 保存训练配置
            config = {
                "agent_type": self.agent_type,
                "env_config": self.env_config,
                "agent_config": self.agent_config,
                "train_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "episodes_completed": self.episodes_completed
            }

            # 自定义JSON序列化器，处理Timestamp类型
            def json_serial(obj):
                """处理无法序列化的对象"""
                if hasattr(obj, 'isoformat'):  # 处理datetime和Timestamp对象
                    return obj.isoformat()
                elif hasattr(obj, 'item'):  # 处理numpy数值类型
                    return obj.item()
                elif isinstance(obj, np.ndarray):  # 处理numpy数组
                    return obj.tolist()
                raise TypeError(f"Type {type(obj)} not serializable")

            with open(f"{save_path}_config.json", 'w') as f:
                json.dump(config, f, indent=4, default=json_serial)

            # 保存评估历史
            with open(f"{save_path}_eval_history.json", 'w') as f:
                json.dump(self.eval_history, f, default=json_serial)

            print(f"模型已保存到: {save_path}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load_model(self, name: str = None):
        """
        加载模型

        参数:
            name: 模型名称 (可选)
        """
        load_name = name or self.model_name

        # 构建加载路径
        model_dir = os.path.join(MODEL_SAVE_PATH, load_name)

        # 检查路径是否存在
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"找不到模型路径: {model_dir}")

        # 加载配置
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

            # 更新配置
            self.agent_type = config['agent_type']
            self.env_config = config['env_config']
            self.agent_config = config['agent_config']
            self.episodes_completed = config['episodes_completed']
            self.best_model_reward = config['best_reward']

            # 根据新配置重新创建环境和代理
            self.train_env = TradingEnv(
                df=self.train_data,
                **self.env_config  # 直接传递完整的环境配置
            )
            self.val_env = TradingEnv(
                df=self.val_data,
                **self.env_config  # 直接传递完整的环境配置
            )
            self._create_agent()

        # 加载代理模型
        self.agent.load(os.path.join(model_dir, "agent"))

        # 加载训练和评估历史
        training_history_path = os.path.join(model_dir, "training_history.json")
        if os.path.exists(training_history_path):
            with open(training_history_path, 'r') as f:
                self.training_history = json.load(f)

        eval_history_path = os.path.join(model_dir, "eval_history.json")
        if os.path.exists(eval_history_path):
            with open(eval_history_path, 'r') as f:
                self.eval_history = json.load(f)

        print(f"成功加载模型: {load_name}")

    def get_training_summary(self) -> Dict:
        """
        获取训练摘要

        返回:
            训练摘要字典
        """
        if not self.training_history:
            return {'status': '未训练'}

        # 获取最新状态
        latest = self.training_history[-1]

        # 计算指标
        if self.eval_history:
            latest_eval = self.eval_history[-1]
            best_eval = max(self.eval_history, key=lambda x: x['avg_return'])
        else:
            latest_eval = {'avg_return': 0, 'win_rate': 0}
            best_eval = {'avg_return': 0, 'win_rate': 0, 'episode': 0}

        return {
            'status': '已完成训练' if not self.is_training else '训练中',
            'agent_type': self.agent_type,
            'episodes_completed': self.episodes_completed,
            'latest_reward': latest['reward'],
            'latest_portfolio_value': latest['portfolio_value'],
            'latest_return': (latest['portfolio_value'] - self.train_env.initial_balance) / self.train_env.initial_balance * 100,
            'latest_eval_return': latest_eval['avg_return'],
            'latest_win_rate': latest_eval['win_rate'],
            'best_eval_return': best_eval['avg_return'],
            'best_eval_episode': best_eval['episode'],
            'training_time': latest['time']
        }

    def train_episode(self, episode, progress_callback=None):
        """
        训练单个回合

        参数:
            episode: 当前回合数
            progress_callback: 可选的进度回调函数

        返回:
            返回格式改为统一的字典格式
        """
        try:
            if not self.is_initialized:
                self.initialize()

            # 打印初始学习率，查看是否能获取
            if hasattr(self.agent, 'get_learning_rate'):
                current_lr = self.agent.get_learning_rate()
                print(f"[学习率检查] 回合 {episode} 开始时学习率: {current_lr:.6f}")

            start_time = time.time()
            # 兼容新版gymnasium接口，reset返回(state, info)元组
            reset_result = self.train_env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) >= 1:
                state = reset_result[0]  # 提取状态
            else:
                # 向后兼容旧接口，直接返回状态
                state = reset_result

            done = False
            truncated = False
            episode_reward = 0
            step = 0
            loss = 0
            action_counts = {0: 0, 1: 0, 2: 0}  # 买入, 持有, 卖出 的计数

            # 确保交易记录为空列表
            self.trade_records = []

            # 记录开始时间
            step_start_time = time.time()

            try:
                # 尝试访问trade_history属性并重置
                self.train_env.trade_history = []
            except AttributeError:
                # 如果不存在就跳过
                pass

            # 初始化变量
            total_reward = 0
            step_reward = 0
            last_action = None
            info = {}
            done = False
            truncated = False
            action_counts = {0: 0, 1: 0, 2: 0}  # 记录动作频率
            consecutive_action_count = {0: 0, 1: 0, 2: 0}  # 记录连续相同动作
            consecutive_zeros_reward = 0  # 记录连续0奖励
            zero_rewards_count = 0  # 0奖励总数

            # 获取训练策略和参数
            train_mode = self.train_mode

            # 初始化交易统计
            self.open_trades = {}  # 当前开放的交易
            self.trade_id_counter = 0  # 交易ID计数器

            # 打印训练开始信息
            print(f"开始训练回合 {self.episodes_completed + 1}, 最大步数: {self.max_steps}, 训练模式: {train_mode}")

            # 用于记录每步的详细信息
            steps_history = []

            # 用于收集训练奖励数据
            rewards_history = []
            portfolio_values_history = []

            # 训练循环
            step_count = 0

            # 设置当前回合的探索率随着步数的进行逐渐降低
            if self.agent_type == 'dqn' and hasattr(self.agent, 'set_epsilon'):
                # 设置初始探索率
                initial_epsilon = min(1.0, 0.5 + (self.force_explore_ratio * 0.5))
                self.agent.set_epsilon(initial_epsilon)
                print(f"当前回合探索率设置为: {initial_epsilon}")

            # 循环执行，直到完成、中止或达到最大步数
            while not done and not truncated and step_count < self.max_steps:
                # 检查是否请求立即停止训练
                if self.stop_immediately:
                    print("检测到立即停止训练请求，中断当前回合训练")
                    break
                # 检查是否请求停止训练
                if self.stop_requested:
                    print("检测到停止训练请求，步数:", step_count)
                    break

                # 更新进度
                if step_count % 100 == 0 or step_count == 1:
                    progress = (step_count / self.max_steps) * 100
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        steps_per_sec = step_count / elapsed_time
                        eta = (self.max_steps - step_count) / steps_per_sec if steps_per_sec > 0 else float('inf')
                        eta_str = f"{eta:.1f}秒" if eta < 120 else f"{eta/60:.1f}分钟"

                        print(f"回合进度: {progress:.1f}% ({step_count}/{self.max_steps}), 速度: {steps_per_sec:.2f}步/秒, 估计剩余时间: {eta_str}")

                    # 根据步数动态调整UI更新频率
                    update_ui = True
                    if step_count > 3000:
                        # 3000步以上，每200步更新一次UI
                        update_ui = (step_count % 200 == 0)
                    elif step_count > 1000:
                        # 1000-3000步，每100步更新一次UI
                        update_ui = True  # 已经由外层if保证
                    else:
                        # 1000步以下，仍然每100步更新
                        update_ui = True  # 已经由外层if保证

                    # 如果连接了UI进度回调，并且满足更新条件，则更新进度
                    if self.progress_callback and update_ui:
                        # 使用train_env内部的数据获取当前环境信息
                        env_info = {
                            'current_step': self.train_env.current_step,
                            'window_size': self.train_env.window_size,
                            'max_episode_steps': self.train_env.max_episode_steps,
                            'price': self.train_env.df.iloc[self.train_env.current_step]['close'] if self.train_env.current_step < len(self.train_env.df) else 0,
                            'balance': self.train_env.balance,
                            'position': self.train_env.position,
                        }

                        # 构建进度数据
                        progress_data = {
                            'step': step_count,
                            'max_steps': self.max_steps,
                            'reward': step_reward if 'step_reward' in locals() else 0,
                            'episode_reward': total_reward,
                            'action_counts': action_counts.copy()
                        }

                        # 添加学习率数据
                        if hasattr(self.agent, 'get_learning_rate'):
                            current_lr = self.agent.get_learning_rate()
                            progress_data['learning_rate'] = current_lr

                            # 如果学习率历史存在且不为空，也添加
                            if hasattr(self, 'learning_rate_history') and self.learning_rate_history:
                                progress_data['learning_rates'] = self.learning_rate_history.copy()
                                # 如果有学习率步数信息，也添加
                                if hasattr(self, 'learning_rate_steps') and len(self.learning_rate_steps) == len(self.learning_rate_history):
                                    progress_data['learning_rate_steps'] = self.learning_rate_steps.copy()

                        # 添加环境信息
                        progress_data['env_info'] = {
                            'current_step': self.train_env.current_step,
                            'window_size': self.train_env.window_size,
                            'max_episode_steps': self.train_env.max_episode_steps,
                            'price': self.train_env.df.iloc[self.train_env.current_step]['close'] if self.train_env.current_step < len(self.train_env.df) else 0,
                            'balance': self.train_env.balance,
                            'position': self.train_env.position,
                        }

                        # 添加奖励历史
                        if rewards_history:
                            progress_data['rewards'] = rewards_history.copy()
                            # 生成奖励曲线的步数信息 - 每10步一个点
                            rewards_steps = []
                            for i in range(len(rewards_history)):
                                if i == 0:
                                    rewards_steps.append(1)  # 第一个点是步数1
                                else:
                                    rewards_steps.append(rewards_steps[i-1] + 10)  # 每10步一个点
                            progress_data['rewards_steps'] = rewards_steps
                            print(f"训练器: 发送奖励曲线步数信息，长度={len(rewards_steps)}")

                        # 添加资产历史
                        if portfolio_values_history:
                            progress_data['returns'] = portfolio_values_history.copy()
                            # 生成收益曲线的步数信息 - 每10步一个点
                            returns_steps = []
                            for i in range(len(portfolio_values_history)):
                                if i == 0:
                                    returns_steps.append(1)  # 第一个点是步数1
                                else:
                                    returns_steps.append(returns_steps[i-1] + 10)  # 每10步一个点
                            progress_data['returns_steps'] = returns_steps

                        # 发送进度更新
                        self.progress_callback(progress_data)

                # 在前400步强制增加随机探索，确保模型尝试不同动作
                force_explore = False
                if step_count < 400 and np.random.random() < 0.5:
                    force_explore = True
                    # 随机选择动作，但要遵守交易规则
                    if self.train_env.position <= 0:
                        # 没有持仓时，只能选择持有或买入
                        action = np.random.choice([1, 2])
                        print(f"DEBUG-EXPLORE: 强制随机探索，无持仓，选择动作: {action}")
                    else:
                        # 有持仓时，可以选择任何动作
                        action = np.random.randint(0, 3)  # 0=卖出, 1=持有, 2=买入
                        print(f"DEBUG-EXPLORE: 强制随机探索，有持仓，选择动作: {action}")
                else:
                    # 正常选择动作
                    action = self.agent.act(state)

                # 跟踪连续相同动作
                if last_action is not None and action == last_action:
                    consecutive_action_count[action] += 1
                    if consecutive_action_count[action] >= 10 and consecutive_action_count[action] % 10 == 0:  # 每10次输出一次日志
                        print(f"DEBUG-ACTION: 连续{consecutive_action_count[action]}次执行动作{action}（0=卖出,1=持有,2=买入）")
                else:
                    # 重置连续动作计数
                    consecutive_action_count = {0: 0, 1: 0, 2: 0}
                    consecutive_action_count[action] = 1

                # 更新动作统计
                action_counts[action] += 1

                # 在训练环境中执行动作
                current_step = self.train_env.current_step
                current_price = self.train_env.df.iloc[current_step]['close']
                position = self.train_env.position
                balance = self.train_env.balance
                position_value = self.train_env.position_value
                portfolio_value = balance + position_value
                position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
                balance_ratio = balance / portfolio_value if portfolio_value > 0 else 0

                # 临时记录当前状态
                pre_step_info = {
                    'step': step_count,
                    'price': current_price,
                    'position': position,
                    'balance': balance,
                    'portfolio_value': portfolio_value,
                    'position_ratio': position_ratio,
                    'action': action
                }

                # 执行步骤
                next_state, reward, done, truncated, info = self.train_env.step(action)

                # 累积奖励
                episode_reward += reward
                step_count += 1

                # 收集奖励和资产数据用于图表 - 每10步收集一次，降低UI绘制的计算压力
                if step_count % 10 == 0 or step_count == 1:  # 第一步和之后每10步
                    rewards_history.append(episode_reward)
                    portfolio_values_history.append(self.train_env.balance + self.train_env.position_value)
                    print(f"DEBUG-DATA-COLLECT: 步数 {step_count}，添加数据点至图表，当前数据点数量: {len(rewards_history)}")

                # 收集交易记录
                if hasattr(self.train_env, 'transaction_history') and self.train_env.transaction_history:
                    # 复制最后一条交易记录
                    for record in self.train_env.transaction_history:
                        if record not in self.trade_records:
                            self.trade_records.append(record)

                # 如果智能体进行了实际的买入或卖出操作，记录下来
                if action in [0, 2]:  # 0=卖出, 2=买入
                    current_time = self.train_env.df.iloc[self.train_env.current_step].name if hasattr(self.train_env.df.iloc[self.train_env.current_step], 'name') else None
                    current_price = self.train_env.df.iloc[self.train_env.current_step]['close']
                    action_type = "买入" if action == 2 else "卖出"
                    trade_record = {
                        'step': step_count,
                        'time': current_time,
                        'action': action_type,
                        'price': current_price,
                        'balance': self.train_env.balance,
                        'position': self.train_env.position,
                        'reward': reward
                    }
                    self.trade_records.append(trade_record)

                # 学习 (如果有足够的样本且未完成)
                if self.agent_type == 'dqn':
                    # 对于DQN，我们存储经验并执行学习
                    if hasattr(self.agent, 'remember'):
                        self.agent.remember(state, action, reward, next_state, done or truncated)

                    # 只有当经验回放缓冲区有足够的样本时才执行学习
                    if hasattr(self.agent, 'memory') and hasattr(self.agent, 'batch_size') and len(self.agent.memory) >= self.agent.batch_size:
                        # 智能调整批量学习步骤，随着步数增加降低学习频率
                        learn_interval = 1  # 默认每步学习一次

                        # 根据步数动态调整学习频率
                        if step_count < 1000:
                            # 前1000步，使用原来的频率
                            if self.train_mode in ['memory_efficient', 'online']:
                                # 在线模式：每步都学习
                                learn_interval = 1
                            elif self.train_mode == 'balanced':
                                # 平衡模式：每3步学习一次
                                learn_interval = 3
                            elif self.train_mode == 'batch':
                                # 批量模式：每10步学习一次
                                learn_interval = 10
                        elif step_count < 3000:
                            # 1000-3000步，降低学习频率
                            if self.train_mode in ['memory_efficient', 'online']:
                                learn_interval = 3
                            elif self.train_mode == 'balanced':
                                learn_interval = 6
                            elif self.train_mode == 'batch':
                                learn_interval = 15
                        else:
                            # 3000步以上，进一步降低学习频率
                            if self.train_mode in ['memory_efficient', 'online']:
                                learn_interval = 5
                            elif self.train_mode == 'balanced':
                                learn_interval = 10
                            elif self.train_mode == 'batch':
                                learn_interval = 20

                        # 根据调整后的间隔执行学习
                        if step_count % learn_interval == 0:
                            loss = self.agent.learn()

                            # 在学习后根据奖励动态调整学习率
                            if hasattr(self.agent, 'get_learning_rate') and hasattr(self.agent, 'set_learning_rate'):
                                # 获取当前学习率
                                current_lr = self.agent.get_learning_rate()

                                # 根据当前步骤的奖励和累积奖励调整学习率
                                # 计算最近几步的平均奖励趋势
                                recent_rewards_window = 5  # 考虑最近5步的奖励
                                if len(rewards_history) >= recent_rewards_window:
                                    recent_rewards = rewards_history[-recent_rewards_window:]
                                    avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
                                    reward_trend = avg_recent_reward - rewards_history[-recent_rewards_window]
                                else:
                                    reward_trend = 0

                                # 如果奖励趋势为负或当前奖励为负，增加学习率以促进探索
                                if reward < -0.01 or reward_trend < -0.05:
                                    # 负奖励或负趋势，增加学习率
                                    if hasattr(self.agent, 'lr_adaptation'):
                                        new_lr = min(current_lr * 1.1, self.agent.lr_adaptation['max_lr'])
                                    else:
                                        new_lr = min(current_lr * 1.1, current_lr * 5.0)
                                    adjustment_reason = "负奖励或负趋势"
                                # 如果奖励趋势为正且当前奖励为正，减小学习率以细化策略
                                elif reward > 0.01 or reward_trend > 0.05:
                                    # 正奖励或正趋势，减小学习率
                                    if hasattr(self.agent, 'lr_adaptation'):
                                        new_lr = max(current_lr * 0.9, self.agent.lr_adaptation['min_lr'])
                                    else:
                                        new_lr = max(current_lr * 0.9, current_lr / 10.0)
                                    adjustment_reason = "正奖励或正趋势"
                                else:
                                    # 奖励接近0，保持学习率不变
                                    new_lr = current_lr
                                    adjustment_reason = "奖励接近0"

                                # 每10步记录学习率并更新UI
                                if step_count % 10 == 0:
                                    # 记录学习率历史和对应的步数
                                    if not hasattr(self, 'learning_rate_history'):
                                        self.learning_rate_history = []
                                        self.learning_rate_steps = []
                                    self.learning_rate_history.append(new_lr)
                                    self.learning_rate_steps.append(step_count)

                                    # 发送更新的学习率历史和步数
                                    if self.progress_callback:
                                        self.progress_callback({
                                            'learning_rate': new_lr,
                                            'learning_rates': self.learning_rate_history.copy(),
                                            'learning_rate_steps': self.learning_rate_steps.copy(),
                                            'step': step_count,
                                            'max_steps': self.max_steps
                                        })
                                    print(f"[学习率更新] 步数: {step_count}, 记录学习率: {new_lr:.6f}，每10步更新一次UI")

                                # 只有当学习率有明显变化时才更新
                                if abs(new_lr - current_lr) > 1e-6:
                                    self.agent.set_learning_rate(new_lr)
                                    # print(f"[步内学习率调整] 步数: {step_count}, 奖励: {reward:.4f}, 趋势: {reward_trend:.4f}, 学习率: {current_lr:.6f} -> {new_lr:.6f}, 原因: {adjustment_reason}")

                                    # 如果有进度回调，只在每10步时发送更新的学习率历史
                                    if self.progress_callback and step_count % 10 == 0:
                                        self.progress_callback({
                                            'learning_rate': new_lr,
                                            'learning_rates': self.learning_rate_history.copy(),
                                            'step': step_count,
                                            'max_steps': self.max_steps
                                        })
                                else:
                                    # 即使学习率没有变化，也每10步更新一次UI
                                    if self.progress_callback and step_count % 10 == 0:
                                        self.progress_callback({
                                            'learning_rate': current_lr,
                                            'step': step_count,
                                            'max_steps': self.max_steps
                                        })
                    elif step_count % 50 == 0:  # 每50步检查一次并打印日志
                        print(f"跳过批量学习：经验回放缓冲区样本不足，当前 {len(self.agent.memory) if hasattr(self.agent, 'memory') else 0}/{self.agent.batch_size if hasattr(self.agent, 'batch_size') else 'unknown'}")

                # 更新状态
                state = next_state

                # 将步骤信息添加到历史记录
                steps_history.append({
                    'step': step_count,
                    'action': action,
                    'reward': reward,
                    'portfolio_value': portfolio_value,
                    'balance': balance,
                    'position': position,
                    'done': done,
                    'truncated': truncated,
                    'price': current_price
                })

                # 更新last_action
                last_action = action

                # 更新info变量
                info.update(pre_step_info)

                # 记录交易信息
                if action != 1:  # 如果不是持有动作
                    self.trade_id_counter += 1
                    trade_id = f"trade_{self.trade_id_counter}"

                    if action == 2:  # 买入
                        trade_record = {
                            'id': trade_id,
                            'type': '买入',
                            'price': current_price,
                            'time': self.train_env.df.index[self.train_env.current_step],
                            'position': self.train_env.position,
                            'step': step_count
                        }
                        self.open_trades[trade_id] = trade_record
                        self.trade_records.append(trade_record)
                    elif action == 0:  # 卖出
                        # 计算卖出时的收益率
                        position_profit_pct = 0.0
                        if hasattr(self.train_env, 'last_buy_price') and self.train_env.last_buy_price > 0:
                            position_profit_pct = (current_price - self.train_env.last_buy_price) / self.train_env.last_buy_price * 100

                        trade_record = {
                            'id': trade_id,
                            'type': '卖出',
                            'price': current_price,
                            'time': self.train_env.df.index[self.train_env.current_step],
                            'position': self.train_env.position,
                            'step': step_count,
                            'profit_pct': position_profit_pct
                        }
                        self.trade_records.append(trade_record)

                # 记录原始环境返回的状态，用于调试
                orig_done = done
                orig_truncated = truncated

            # 计算平均每步奖励
            avg_reward = total_reward / step_count if step_count > 0 else 0

            # 计算动作分布
            total_actions = sum(action_counts.values())
            action_distribution = {
                action: count / total_actions * 100 if total_actions > 0 else 0
                for action, count in action_counts.items()
            }

            # 训练结束，计算统计数据
            training_time = time.time() - start_time
            steps_per_second = step_count / training_time if training_time > 0 else 0

            # 获取最终状态
            final_state = {
                'balance': self.train_env.balance,
                'position': self.train_env.position,
                'portfolio_value': self.train_env.balance + self.train_env.position_value,
                'initial_value': self.train_env.initial_balance,
                'return_pct': ((self.train_env.balance + self.train_env.position_value) / self.train_env.initial_balance - 1) * 100,
                'max_drawdown': self.train_env.max_drawdown * 100,
                'trade_count': self.train_env.trade_count,
                'fees_paid': self.train_env.fees_paid if hasattr(self.train_env, 'fees_paid') else 0
            }

            # 更新训练统计
            self.episodes_completed += 1
            self.total_steps_completed += step_count

            # 根据收益更新学习率（如果智能体支持）
            if self.agent_type == 'dqn' and hasattr(self.agent, 'update_performance'):
                # 计算回合收益率
                episode_return = final_state['return_pct']

                # 记录更新前的学习率
                if hasattr(self.agent, 'get_learning_rate'):
                    current_lr = self.agent.get_learning_rate()
                    print(f"[学习率检查] 回合 {episode} 更新前学习率: {current_lr:.6f}, 收益率: {episode_return:.2f}%")

                # 更新代理的学习率（基于性能）
                update_result = self.agent.update_performance(episode_return, total_reward)

                # 记录更新后的学习率
                if hasattr(self.agent, 'get_learning_rate'):
                    new_lr = self.agent.get_learning_rate()
                    print(f"[学习率检查] 回合 {episode} 更新后学习率: {new_lr:.6f}, 变化: {new_lr-current_lr:.6f}")

                    # 确保学习率历史存在
                    if not hasattr(self, 'learning_rate_history'):
                        self.learning_rate_history = []
                    # 添加最终学习率到历史
                    self.learning_rate_history.append(new_lr)
                    print(f"[学习率采样] 回合结束时添加学习率: {new_lr:.6f}")

            # 收集和存储学习率历史
            if not hasattr(self, 'learning_rate_history'):
                self.learning_rate_history = []

            # 获取当前学习率（如果代理支持）
            current_lr = 0.0
            if hasattr(self.agent, 'get_learning_rate'):
                current_lr = self.agent.get_learning_rate()
                print(f"[学习率检查] 回合 {episode} 结束时学习率: {current_lr:.6f}")

            # 更新智能体的探索率
            if self.agent_type == 'dqn' and hasattr(self.agent, 'epsilon'):
                # 探索率随着训练降低
                epsilon_decay = 0.995  # 探索率的衰减系数

                # 确保探索率不会太低，保持一定的探索能力
                min_epsilon = 0.01

                # 更新探索率
                new_epsilon = max(min_epsilon, self.agent.epsilon * epsilon_decay)
                self.agent.epsilon = new_epsilon

                print(f"探索率更新: {self.agent.epsilon:.4f} -> {new_epsilon:.4f}")

            # 总结统计，输出训练结果
            print(f"\n===== 训练回合 {self.episodes_completed} 完成 =====")
            print(f"步数: {step_count}")
            print(f"总奖励: {total_reward:.2f}, 平均每步奖励: {avg_reward:.4f}")
            print(f"训练时间: {training_time:.2f}秒, 速度: {steps_per_second:.2f}步/秒")

            # 显示动作分布
            print("动作分布:")
            for action, percentage in action_distribution.items():
                action_name = ['卖出', '持有', '买入'][action]
                print(f"  {action_name}: {percentage:.1f}% ({action_counts[action]}次)")

            # 显示奖励为0的比例
            zero_reward_pct = zero_rewards_count / step_count * 100 if step_count > 0 else 0
            print(f"奖励为0的步数: {zero_rewards_count}/{step_count} ({zero_reward_pct:.1f}%)")

            # 显示最终账户状态
            print(f"初始资金: {final_state['initial_value']:.2f}")
            print(f"最终资金: {final_state['balance']:.2f}")
            print(f"持仓价值: {final_state['portfolio_value'] - final_state['balance']:.2f}")
            print(f"总资产: {final_state['portfolio_value']:.2f}")
            print(f"收益率: {final_state['return_pct']:.2f}%")
            print(f"最大回撤: {final_state['max_drawdown']:.2f}%")
            print(f"交易次数: {final_state['trade_count']}")
            print(f"支付手续费: {final_state['fees_paid']:.2f}")

            # 构建最终结果
            result = {
                'success': True,
                'episode': self.episodes_completed,
                'reward': total_reward,
                'portfolio_value': self.train_env.balance + self.train_env.position_value,
                'return': (self.train_env.balance + self.train_env.position_value - self.train_env.initial_balance) / self.train_env.initial_balance * 100,
                'drawdown': self.train_env.max_drawdown * 100 if hasattr(self.train_env, 'max_drawdown') else 0,
                'trade_count': self.train_env.trade_count if hasattr(self.train_env, 'trade_count') else 0,
                'action_counts': action_counts,
                'zero_rewards': zero_rewards_count,
                'steps': step_count,
                'elapsed_time': time.time() - step_start_time if 'step_start_time' in locals() else 0,
                'steps_per_second': step_count / (time.time() - step_start_time) if (time.time() - step_start_time) > 0 else 0,
                'balance': self.train_env.balance,
                'position_value': self.train_env.position_value,
                'position': self.train_env.position,
                'max_return': self.train_env.max_return * 100 if hasattr(self.train_env, 'max_return') else 0,
                'learning_rate': current_lr
            }

            # 记录训练历史
            self.training_history.append(result)

            # 发送最终进度更新（无论步数多少，确保UI获得完整数据）
            if self.progress_callback:
                # 复制结果并添加最终标记
                final_update = result.copy()
                final_update['is_done'] = True  # 标记为最终更新
                final_update['rewards'] = rewards_history  # 添加完整奖励历史
                final_update['returns'] = portfolio_values_history  # 添加完整资产历史

                # 生成并添加收益曲线的步数信息
                # 收益曲线数据是每10步采样一次
                returns_steps = []
                for i in range(len(portfolio_values_history)):
                    if i == 0:
                        returns_steps.append(1)  # 第一个点是步数1
                    else:
                        returns_steps.append(returns_steps[i-1] + 10)  # 每10步一个点
                final_update['returns_steps'] = returns_steps
                print(f"训练器: 发送收益曲线步数信息，长度={len(returns_steps)}")

                # 确保学习率数据被正确包含
                if hasattr(self, 'learning_rate_history') and self.learning_rate_history:
                    final_update['learning_rates'] = self.learning_rate_history.copy()
                    # 如果有学习率步数信息，也发送
                    if hasattr(self, 'learning_rate_steps') and len(self.learning_rate_steps) == len(self.learning_rate_history):
                        final_update['learning_rate_steps'] = self.learning_rate_steps.copy()
                        print(f"训练器: 发送学习率历史和步数，长度={len(self.learning_rate_history)}，最新值={self.learning_rate_history[-1]:.6f}")
                    else:
                        print(f"训练器: 发送学习率历史（无步数），长度={len(self.learning_rate_history)}，最新值={self.learning_rate_history[-1]:.6f}")
                elif current_lr > 0:
                    # 如果没有历史但有当前值，也发送单个值
                    final_update['learning_rates'] = [current_lr]
                    final_update['learning_rate'] = current_lr
                    final_update['learning_rate_steps'] = [step_count]  # 添加当前步数
                    print(f"训练器: 发送单个学习率值 {current_lr:.6f}, 步数={step_count}")

                final_update['env_info'] = {
                    'current_step': self.train_env.current_step,
                    'window_size': self.train_env.window_size,
                    'max_episode_steps': self.train_env.max_episode_steps,
                    'price': self.train_env.df.iloc[self.train_env.current_step]['close'] if self.train_env.current_step < len(self.train_env.df) else 0,
                    'balance': self.train_env.balance,
                    'position': self.train_env.position,
                }
                self.progress_callback(final_update)

            print(f"回合 {self.episodes_completed} 完成 - 回报: {total_reward:.4f}, 组合价值: {result['portfolio_value']:.2f}")
            print(f"动作分布: 卖出={action_counts[0]}, 持有={action_counts[1]}, 买入={action_counts[2]}")
            print(f"训练状态: 回合={self.episodes_completed}, 最大收益率={result['max_return']:.2f}%, 最大回撤={result['drawdown']:.2f}%")

            # 仅返回结果字典，不再使用(True, result)的元组形式
            return result

        except Exception as e:
            print(f"训练回合出错: {str(e)}")
            import traceback
            traceback.print_exc()

            # 构建错误结果字典
            error_result = {
                'episode': getattr(self, 'episodes_completed', episode),
                'success': False,
                'error': str(e),
                'action_counts': action_counts if 'action_counts' in locals() else {0: 0, 1: 0, 2: 0},
                'steps': step if 'step' in locals() else 0,
                'time': time.time() - start_time if 'start_time' in locals() else 0
            }

            return error_result

    def update_top_models(self, score, weights, episode):
        """
        更新顶级模型列表

        参数:
            score: 模型得分
            weights: 模型权重
            episode: 训练回合
        """
        # 添加当前模型
        self.top_models.append((score, weights, episode))

        # 按得分降序排序
        self.top_models.sort(key=lambda x: x[0], reverse=True)

        # 只保留前N个
        if len(self.top_models) > self.max_top_models:
            self.top_models = self.top_models[:self.max_top_models]

        # 打印当前top模型情况
        print(f"[训练] 当前Top{min(3, len(self.top_models))}模型:")
        for i, (s, _, ep) in enumerate(self.top_models[:min(3, len(self.top_models))]):
            print(f"  第{i+1}名: 回合{ep}, 得分{s:.4f}")

    def get_best_model_weights(self):
        """
        获取最佳模型权重

        返回:
            最佳模型权重，如果没有则返回None
        """
        if self.best_model_weights is not None:
            return self.best_model_weights
        elif len(self.top_models) > 0:
            # 返回得分最高的模型权重
            return self.top_models[0][1]
        elif hasattr(self, 'agent') and self.agent:
            # 如果没有找到更好的模型，返回当前模型权重
            return self.agent.get_weights() if hasattr(self.agent, 'get_weights') else None
        return None