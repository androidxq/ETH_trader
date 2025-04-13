"""
强化学习训练线程
"""

import time
import traceback
from typing import Dict
from PyQt6.QtCore import QThread, pyqtSignal
import pandas as pd
import numpy as np
import json
from datetime import datetime

# 导入训练器
from rl_strategies.trainer import RLTrainer


class RLTrainingThread(QThread):
    """强化学习训练线程"""
    
    # 信号定义
    progress_signal = pyqtSignal(dict)  # 训练进度信号
    eval_signal = pyqtSignal(dict)      # 评估结果信号
    complete_signal = pyqtSignal()      # 完成信号
    error_signal = pyqtSignal(str)      # 错误信号
    log_signal = pyqtSignal(str)        # 日志信号
    epsilon_signal = pyqtSignal(str)    # 探索率信号
    
    def __init__(self, trainer: RLTrainer = None, max_episodes: int = 500, progress_callback=None, env_config=None, train_config=None, train_df=None, eval_df=None, load_model_path=None, save_model_path=None):
        """
        初始化训练线程
        
        参数:
            trainer: 训练器对象
            max_episodes: 最大训练回合数
            progress_callback: 进度回调函数
            env_config: 环境配置参数
            train_config: 训练配置参数
            train_df: 训练数据
            eval_df: 评估数据
            load_model_path: 加载模型路径
            save_model_path: 保存模型路径
        """
        super().__init__()
        self.trainer = trainer
        self.max_episodes = max_episodes
        self.progress_callback = progress_callback
        self.stop_requested = False
        self.env_config = env_config or {}
        self.train_config = train_config or {}
        self.train_df = train_df
        self.eval_df = eval_df
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.running = False
        self.agent = None
        self.train_env = None
        self.eval_env = None
        
        # 为trainer设置progress_callback，修正条件判断逻辑
        if self.trainer is not None:
            if not hasattr(self.trainer, 'progress_callback') or self.trainer.progress_callback is None:
                print("DEBUG: 在初始化时设置trainer的progress_callback")
                # 使用lambda避免方法名冲突
                self.trainer.progress_callback = lambda data: self.process_progress_data(data)
        else:
            print("DEBUG: trainer为None，将在run方法中创建")
    
    def run(self):
        """执行强化学习训练任务"""
        try:
            self.running = True
            
            # 检查trainer是否已存在，如果存在则使用已有trainer
            if self.trainer:
                print("DEBUG: 使用已存在的训练器")
                # 确保回调函数已设置
                if not hasattr(self.trainer, 'progress_callback') or self.trainer.progress_callback is None:
                    self.trainer.progress_callback = lambda data: self.process_progress_data(data)
                
                # 发送训练开始信号
                self.progress_signal.emit({"status": "started", "max_episodes": self.max_episodes})
                
                # 开始训练
                self.trainer.train(episodes=self.max_episodes, eval_interval=20)
                
                # 在训练结束后检查是否是因为停止请求而结束
                if self.stop_requested:
                    self.log_signal.emit("训练已按请求停止")
                    print("DEBUG: 训练已按请求停止")
                else:
                    self.log_signal.emit("训练已完成所有轮次")
                    print("DEBUG: 训练已完成所有轮次")
                
                # 训练完成后评估并保存模型
                if not self.stop_requested:
                    # 获取最佳模型权重
                    if hasattr(self.trainer, 'get_best_model_weights'):
                        best_weights = self.trainer.get_best_model_weights()
                    else:
                        # 如果方法不存在，使用当前权重作为最佳权重
                        print("警告: trainer对象没有get_best_model_weights方法，使用当前权重")
                        best_weights = self.agent.get_weights() if hasattr(self.agent, 'get_weights') else None
                    
                    if best_weights is not None:
                        # 应用最佳权重并保存
                        self.agent.set_weights(best_weights)
                        
                        # 保存模型
                        if self.save_model_path:
                            self.trainer.save_model(self.save_model_path)
                            print(f"已保存最佳模型到: {self.save_model_path}")
                    else:
                        print("WARNING: 未找到有效的最佳模型权重")
                
                # 发出训练完成信号
                self.complete_signal.emit()
                return
            
            # 如果没有现有trainer，创建新的训练环境和trainer
            try:
                # 获取各项配置
                print("DEBUG: 开始准备训练环境和参数")
                
                # 环境配置
                env_config = self.env_config
                
                # 确保max_episode_steps是设置的合理值
                if 'max_episode_steps' not in env_config or env_config['max_episode_steps'] <= 0:
                    env_config['max_episode_steps'] = 500  # 默认使用500步
                    print(f"WARNING: 未指定max_episode_steps或值不合理，设置为默认值 {env_config['max_episode_steps']}")
                
                # 训练配置
                train_episodes = self.train_config.get('episodes', 500)
                batch_size = self.train_config.get('batch_size', 32)
                learning_rate = self.train_config.get('learning_rate', 0.001)
                discount_factor = self.train_config.get('discount_factor', 0.99)
                update_target_every = self.train_config.get('update_target_every', 5)
                agent_type = self.train_config.get('agent_type', 'dqn')
                verbose = self.train_config.get('verbose', True)  # 添加verbose参数
                
                print(f"DEBUG: 训练配置 - 回合数: {train_episodes}, 学习率: {learning_rate}, 智能体类型: {agent_type}")
                print(f"DEBUG: 环境配置 - max_episode_steps: {env_config['max_episode_steps']}, 窗口大小: {env_config['window_size']}")
                
                # 区分奖励权重和环境配置，确保不会传入不支持的参数
                reward_weights = None
                if 'reward_weights' in env_config:
                    reward_weights = env_config.pop('reward_weights')
                    print(f"DEBUG: 奖励权重配置: {reward_weights}")
                
                # 确保数据已加载
                if self.train_df is None or self.eval_df is None:
                    raise ValueError("训练和评估数据未加载，无法开始训练")
                
                # 生成训练和评估环境，明确指定环境类型
                from rl_strategies.environments.trading_env import TradingEnv
                print(f"DEBUG: 创建训练环境，数据形状: {self.train_df.shape}")
                self.train_env = TradingEnv(df=self.train_df, env_type='training', **env_config)
                
                print(f"DEBUG: 创建评估环境，数据形状: {self.eval_df.shape}")
                self.eval_env = TradingEnv(df=self.eval_df, env_type='evaluation', **env_config)
                
                # 设置环境的reward_weights (如果使用复合奖励)
                if reward_weights and env_config.get('reward_type') == 'compound':
                    self.train_env.reward_weights = reward_weights
                    self.eval_env.reward_weights = reward_weights
                    print(f"DEBUG: 已设置复合奖励权重")
                    
                # 打印环境信息以确认初始化正确
                print(f"DEBUG: 训练环境初始化 - 数据长度: {len(self.train_env.df)}, 最大步数: {self.train_env.max_episode_steps}")
                print(f"DEBUG: 评估环境初始化 - 数据长度: {len(self.eval_env.df)}, 最大步数: {self.eval_env.max_episode_steps}")

                # 创建智能体
                from rl_strategies.agents.dqn_agent import DQNAgent
                
                # 获取状态和动作空间维度
                state_size = self.train_env.observation_space.shape[0]
                action_size = self.train_env.action_space.n
                
                print(f"DEBUG: 状态空间维度: {state_size}, 动作空间维度: {action_size}")
                
                # 创建DQN智能体
                agent_config = {
                    'batch_size': batch_size,
                    'gamma': discount_factor,
                    'learning_rate': learning_rate,
                    'update_target_every': update_target_every,
                    'max_learning_rate': self.train_config.get('max_learning_rate'),
                    'lr_adaptation': self.train_config.get('lr_adaptation', {}),
                    'epsilon_start': 1.0,  # 确保设置初始探索率
                    'epsilon_end': 0.01,   # 确保设置最终探索率
                    'epsilon_decay': 0.995 # 确保设置探索率衰减因子
                }

                # 打印完整的代理配置
                print("\n=== 创建DQN代理的完整配置 ===")
                print(f"基础学习率: {agent_config['learning_rate']}")
                print(f"最大学习率: {agent_config['max_learning_rate']}")
                print(f"学习率自适应配置: {agent_config['lr_adaptation']}")
                print("="*50)

                self.agent = DQNAgent(
                    state_dim=state_size,
                    action_dim=action_size,
                    config=agent_config
                )
                
                # 设置探索率回调函数
                if hasattr(self, 'setup_agent_callback') and callable(self.setup_agent_callback):
                    self.setup_agent_callback(self.agent)
                    print("DEBUG: 已通过回调设置探索率监控")
                else:
                    print("警告: 未找到setup_agent_callback函数，无法设置探索率监控")
                    # 尝试直接设置epsilon_callback
                    if hasattr(self.agent, 'register_epsilon_callback'):
                        def epsilon_callback(message):
                            print(f"探索率更新: {message}")
                            # 发送探索率信息到UI专门的探索率信号
                            self.epsilon_signal.emit(message)
                        
                        self.agent.register_epsilon_callback(epsilon_callback)
                        print("已直接注册探索率回调函数")
                
                # 检查是否有上一次训练的模型权重需要加载
                if self.load_model_path:
                    try:
                        self.agent.load_weights(self.load_model_path)
                        print(f"DEBUG: 已加载预训练模型: {self.load_model_path}")
                    except Exception as e:
                        print(f"警告: 无法加载模型权重: {e}")
                
                # 创建训练器
                from rl_strategies.trainer import RLTrainer
                
                # 创建完整的代理配置
                complete_agent_config = {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'discount_factor': discount_factor,
                    'update_target_every': update_target_every,
                    'verbose': verbose,
                    'max_learning_rate': self.train_config.get('max_learning_rate'),
                    'lr_adaptation': self.train_config.get('lr_adaptation', {})
                }

                # 打印完整的代理配置
                print("\n=== 创建训练器时的完整代理配置 ===")
                print(f"基础学习率: {complete_agent_config['learning_rate']}")
                print(f"最大学习率: {complete_agent_config['max_learning_rate']}")
                print(f"学习率自适应配置: {complete_agent_config['lr_adaptation']}")
                print("="*50)
                
                self.trainer = RLTrainer(
                    agent_type=agent_type,
                    train_data=self.train_df,
                    val_data=self.eval_df,
                    env_config=env_config,
                    agent_config=complete_agent_config
                )
                
                # 确保训练器的环境已正确设置环境类型
                self.trainer.train_env.env_type = 'training'
                self.trainer.val_env.env_type = 'evaluation'
                
                # 初始化交易记录存储
                self.trainer.train_env.train_transaction_history = []
                self.trainer.val_env.eval_transaction_history = []
                
                # 设置训练器的progress_callback
                # 使用内部处理方法，避免命名冲突
                self.trainer.progress_callback = lambda data: self.process_progress_data(data)
                
                # 设置训练状态和信号
                self.progress_signal.emit({"status": "started", "max_episodes": train_episodes})
                
                # 开始训练
                print(f"DEBUG: 开始训练，总回合数: {train_episodes}")
                self.trainer.train(episodes=train_episodes, eval_interval=20, verbose=verbose)
                
                # 在训练结束后检查是否是因为停止请求而结束
                if self.stop_requested:
                    self.log_signal.emit("训练已按请求停止")
                    print("DEBUG: 训练已按请求停止")
                else:
                    self.log_signal.emit("训练已完成所有轮次")
                    print("DEBUG: 训练已完成所有轮次")
                
                # 训练完成后评估并保存模型
                if not self.stop_requested:
                    # 获取最佳模型权重
                    if hasattr(self.trainer, 'get_best_model_weights'):
                        best_weights = self.trainer.get_best_model_weights()
                    else:
                        # 如果方法不存在，使用当前权重作为最佳权重
                        print("警告: trainer对象没有get_best_model_weights方法，使用当前权重")
                        best_weights = self.agent.get_weights() if hasattr(self.agent, 'get_weights') else None
                    
                    if best_weights is not None:
                        # 应用最佳权重并保存
                        self.agent.set_weights(best_weights)
                        
                        # 保存模型
                        if self.save_model_path:
                            self.trainer.save_model(self.save_model_path)
                            print(f"已保存最佳模型到: {self.save_model_path}")
                    else:
                        print("WARNING: 未找到有效的最佳模型权重")
                
                # 发出训练完成信号
                self.complete_signal.emit()
                
            except Exception as e:
                import traceback
                error_msg = f"训练环境初始化或训练过程出错: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                self.error_signal.emit(error_msg)
        
        except Exception as e:
            import traceback
            error_msg = f"训练过程出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error_signal.emit(error_msg)
        
        finally:
            self.running = False

    def process_progress_data(self, data: Dict):
        """
        处理训练进度回调
        
        参数:
            data: 包含训练状态的字典
        """
        try:
            # 从训练进度数据复制所需字段
            progress_data = {}
            
            # 确认这是训练数据
            progress_data['source_type'] = 'training'
            
            # 复制基本字段
            for key in ['step', 'max_steps', 'reward', 'episode_reward', 'action_counts', 'env_info', 'rewards', 'returns']:
                if key in data:
                    progress_data[key] = data[key]
            
            # 复制学习率相关字段
            for key in ['learning_rate', 'learning_rates']:
                if key in data:
                    progress_data[key] = data[key]
                    print(f"DEBUG: 复制学习率数据 {key}, 值: {data[key] if key == 'learning_rate' else f'列表长度={len(data[key])}'}")
            
            # 如果有学习率历史数据，但没有直接传递，从训练器获取
            if 'learning_rate' in data and 'learning_rates' not in data and hasattr(self.trainer, 'learning_rate_history'):
                progress_data['learning_rates'] = self.trainer.learning_rate_history.copy()
                print(f"DEBUG: 从训练器获取学习率历史，长度={len(progress_data['learning_rates'])}")
            
            # 添加当前回合和总回合数信息
            progress_data['episode'] = self.trainer.episodes_completed if hasattr(self.trainer, 'episodes_completed') else 0
            progress_data['max_episodes'] = self.train_config.get('episodes', 500)
            
            # 确保最大步数信息存在，如果不存在则使用环境配置
            if 'max_steps' not in progress_data and hasattr(self.trainer, 'train_env'):
                progress_data['max_steps'] = self.trainer.train_env.max_episode_steps
            
            # 添加收益率信息（如果可用）
            if 'env_info' in data and 'balance' in data['env_info'] and hasattr(self.trainer, 'train_env'):
                initial_balance = self.trainer.train_env.initial_balance
                current_balance = data['env_info']['balance']
                position_value = self.trainer.train_env.position_value if hasattr(self.trainer.train_env, 'position_value') else 0
                total_value = current_balance + position_value
                return_pct = (total_value / initial_balance - 1) * 100
                progress_data['return'] = return_pct
            
            # 获取交易记录 - 确保使用训练环境的交易记录
            if hasattr(self.trainer, 'train_env') and hasattr(self.trainer.train_env, 'transaction_history'):
                # 复制完整交易记录
                progress_data['trade_records'] = self.trainer.train_env.transaction_history.copy()
                print(f"从train_env获取到 {len(progress_data['trade_records'])} 条交易记录")
            elif 'trade_records' in data:
                progress_data['trade_records'] = data['trade_records']
                print(f"从data中获取到 {len(progress_data['trade_records'])} 条交易记录")
            else:
                progress_data['trade_records'] = []
                print("未找到交易记录")
            
            # 计算当前进度百分比
            if 'episode' in data and hasattr(self, 'max_episodes'):
                progress_data['progress_pct'] = (data['episode'] / self.max_episodes) * 100
            
            # 添加交易环境信息
            if 'env_info' in data and isinstance(data['env_info'], dict):
                env_info = data['env_info']
                
                # 计算收益率
                if 'balance' in env_info and 'position' in env_info:
                    current_balance = env_info['balance']
                    current_position = env_info['position']
                    current_price = env_info.get('price', 0)
                    
                    # 计算总资产
                    total_value = current_balance + (current_position * current_price)
                    
                    # 获取初始资金
                    initial_balance = self.env_config.get('initial_balance', 10000)
                    
                    # 计算收益率
                    return_pct = ((total_value / initial_balance) - 1) * 100
                    progress_data['return'] = return_pct
                    progress_data['portfolio_value'] = total_value
            
            # 发出进度信号
            self.progress_signal.emit(progress_data)
            
            # 如果有步骤信息，输出日志
            if 'step' in data and 'max_steps' in data:
                step_progress = (data['step'] / data['max_steps']) * 100 if data['max_steps'] > 0 else 0
                
                # 构建日志消息
                log_msg = f"步数: {data['step']}/{data['max_steps']}"
                
                # 如果有资产信息，添加到日志消息
                if 'env_info' in data and 'balance' in data['env_info']:
                    portfolio_value = data['env_info']['balance']
                    if 'position' in data['env_info'] and 'price' in data['env_info']:
                        portfolio_value += data['env_info']['position'] * data['env_info']['price']
                    log_msg += f" - 资产: {portfolio_value:.2f}"
                
                # 发送日志消息
                self.log_signal.emit(log_msg)
                
            # 如果有交易记录，在日志中显示
            if 'trade_records' in progress_data and progress_data['trade_records']:
                self.log_signal.emit(f"当前回合有 {len(progress_data['trade_records'])} 条交易记录")
                
        except Exception as e:
            self.log_signal.emit(f"处理训练进度回调时出错: {str(e)}\n{traceback.format_exc()}")
    
    def eval_callback(self, data: Dict):
        """评估结果回调"""
        try:
            # 添加回合信息
            data['episode'] = self.trainer.episodes_completed
            
            # 标记为评估数据
            data['source_type'] = 'evaluation'
            
            # 确保trade_history是一个列表，并且使用评估环境的交易记录
            if hasattr(self.trainer, 'val_env') and hasattr(self.trainer.val_env, 'transaction_history'):
                # 使用评估环境的交易记录
                eval_trades = self.trainer.val_env.transaction_history.copy()
                if len(eval_trades) > 0:
                    print(f"DEBUG - 评估回调: 从val_env获取到 {len(eval_trades)} 条交易记录")
                    # 使用统一的字段名
                    data['trade_history'] = eval_trades
                    data['trades'] = eval_trades  # 为兼容性保留
                else:
                    print("DEBUG - 评估回调: val_env中没有交易记录")
                    data['trade_history'] = []
                    data['trades'] = []
            elif 'trade_history' in data:
                print(f"DEBUG - 评估回调: 收到交易记录 {len(data['trade_history'])} 条")
            elif 'trades' in data:
                print(f"DEBUG - 评估回调: 收到trades交易记录 {len(data['trades'])} 条")
                # 将trades复制到trade_history，保持兼容性
                data['trade_history'] = data['trades']
            else:
                print("DEBUG - 评估回调: 未收到任何交易记录")
                # 如果没有交易记录，初始化为空列表
                data['trade_history'] = []
                data['trades'] = []
            
            # 将评估结果保存到JSON文件
            try:
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
                
                # 创建一个可序列化的副本
                save_data = {}
                for key, value in data.items():
                    if key == 'trade_history' or key == 'trades':
                        # 保留所有原始字段，不再过滤字段
                        save_data[key] = []
                        for trade in value:
                            # 复制原始交易记录的所有字段
                            trade_copy = dict(trade)
                            # 确保关键字段存在且类型正确
                            if 'timestamp' in trade:
                                trade_copy['timestamp'] = str(trade['timestamp'])
                            if 'price' in trade:
                                trade_copy['price'] = float(trade['price'])
                            if 'amount' in trade:
                                trade_copy['amount'] = float(trade['amount'])
                            if 'profit' in trade:
                                trade_copy['profit'] = float(trade['profit'])
                            if 'balance' in trade:
                                trade_copy['balance'] = float(trade['balance'])
                            # 添加完整交易记录
                            save_data[key].append(trade_copy)
                            
                        # 打印调试信息
                        if len(save_data[key]) > 0:
                            print(f"DEBUG - 交易记录示例: {save_data[key][0]}")
                    else:
                        # 直接复制其他字段
                        save_data[key] = value
                
                # 保存到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"eval_result_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2, default=json_serial)
                print(f"DEBUG - 评估回调: 已保存评估结果到 {filename}")
            except Exception as e:
                print(f"保存评估结果时出错: {str(e)}")
            
            # 发出评估信号
            self.eval_signal.emit(data)
            
            # 输出评估日志
            self.log_signal.emit(f"评估结果 (回合 {data['episode']}): "
                               f"平均奖励: {data['avg_reward']:.4f}, "
                               f"平均收益率: {data['avg_return']:.2f}%, "
                               f"胜率: {data['win_rate']:.2f}%")
                               
        except Exception as e:
            error_msg = f"处理评估回调时出错: {str(e)}\n{traceback.format_exc()}"
            print(f"ERROR: {error_msg}")
            self.log_signal.emit(f"评估回调错误: {str(e)}")
    
    def stop(self):
        """停止训练"""
        if self.stop_requested:
            # 已经请求停止，避免重复操作
            print("DEBUG: 已经请求停止训练，请等待轮次结束")
            self.log_signal.emit("已经请求停止训练，请等待当前轮次结束...")
            return
            
        # 设置停止标志
        self.stop_requested = True
        print("DEBUG: 设置训练线程停止标志 stop_requested=True")
        self.log_signal.emit("训练将在当前轮次结束后停止...")
        
        # 确保停止信号传递到trainer对象
        if self.trainer:
            try:
                # 设置trainer的停止标志
                self.trainer.stop_requested = True
                print("DEBUG: 设置trainer停止标志 stop_requested=True")
                
                # 显式调用trainer的stop_training方法
                try:
                    stop_result = self.trainer.stop_training()
                    print(f"DEBUG: 成功调用trainer.stop_training()，结果: {stop_result}")
                except Exception as e:
                    print(f"WARNING: 调用trainer.stop_training()时出错: {str(e)}")
                    # 出错时重试一次设置停止标志
                    try:
                        self.trainer.stop_requested = True
                        print("DEBUG: 重试设置trainer停止标志 stop_requested=True")
                    except:
                        pass
                
                # 确认停止标志已设置
                if hasattr(self.trainer, 'stop_requested') and self.trainer.stop_requested:
                    self.log_signal.emit("已向训练器发送停止信号，训练将在当前轮次结束后停止")
                else:
                    self.log_signal.emit("警告: 可能无法设置训练器停止标志，训练可能不会立即停止")
            except Exception as e:
                print(f"ERROR: 停止训练时出错: {str(e)}")
                self.log_signal.emit(f"停止训练时出错: {str(e)}")
        else:
            self.log_signal.emit("警告: 无法找到有效的训练器对象")

    def start_training(self):
        """开始训练过程"""
        try:
            self.running = True
            
            # 获取各项配置
            print("DEBUG: 开始准备训练环境和参数")
            
            # 环境配置
            env_config = self.env_config
            
            # 确保max_episode_steps是设置的合理值
            if 'max_episode_steps' not in env_config or env_config['max_episode_steps'] <= 0:
                env_config['max_episode_steps'] = 500  # 默认使用500步
                print(f"WARNING: 未指定max_episode_steps或值不合理，设置为默认值 {env_config['max_episode_steps']}")
            
            # 训练配置
            train_episodes = self.train_config.get('episodes', 500)
            batch_size = self.train_config.get('batch_size', 32)
            learning_rate = self.train_config.get('learning_rate', 0.001)
            discount_factor = self.train_config.get('discount_factor', 0.99)
            update_target_every = self.train_config.get('update_target_every', 5)
            agent_type = self.train_config.get('agent_type', 'dqn')
            verbose = self.train_config.get('verbose', True)  # 添加verbose参数
            
            print(f"DEBUG: 训练配置 - 回合数: {train_episodes}, 学习率: {learning_rate}, 智能体类型: {agent_type}")
            print(f"DEBUG: 环境配置 - max_episode_steps: {env_config['max_episode_steps']}, 窗口大小: {env_config['window_size']}")
            
            # 区分奖励权重和环境配置，确保不会传入不支持的参数
            reward_weights = None
            if 'reward_weights' in env_config:
                reward_weights = env_config.pop('reward_weights')
                print(f"DEBUG: 奖励权重配置: {reward_weights}")
            
            # 确保数据已加载
            if self.train_df is None or self.eval_df is None:
                raise ValueError("训练和评估数据未加载，无法开始训练")
            
            # 生成训练和评估环境
            from rl_strategies.environments.trading_env import TradingEnv
            print(f"DEBUG: 创建训练环境，数据形状: {self.train_df.shape}")
            self.train_env = TradingEnv(df=self.train_df, **env_config)
            
            print(f"DEBUG: 创建评估环境，数据形状: {self.eval_df.shape}")
            self.eval_env = TradingEnv(df=self.eval_df, **env_config)
            
            # 设置环境的reward_weights (如果使用复合奖励)
            if reward_weights and env_config.get('reward_type') == 'compound':
                self.train_env.reward_weights = reward_weights
                self.eval_env.reward_weights = reward_weights
                print(f"DEBUG: 已设置复合奖励权重")
                
            # 打印环境信息以确认初始化正确
            print(f"DEBUG: 训练环境初始化 - 数据长度: {len(self.train_env.df)}, 最大步数: {self.train_env.max_episode_steps}")
            print(f"DEBUG: 评估环境初始化 - 数据长度: {len(self.eval_env.df)}, 最大步数: {self.eval_env.max_episode_steps}")

            # 创建智能体
            from rl_strategies.agents.dqn_agent import DQNAgent
            
            # 获取状态和动作空间维度
            state_size = self.train_env.observation_space.shape[0]
            action_size = self.train_env.action_space.n
            
            print(f"DEBUG: 状态空间维度: {state_size}, 动作空间维度: {action_size}")
            
            # 创建DQN智能体
            agent_config = {
                'batch_size': batch_size,
                'gamma': discount_factor,
                'learning_rate': learning_rate,
                'update_target_every': update_target_every,
                'max_learning_rate': self.train_config.get('max_learning_rate'),
                'lr_adaptation': self.train_config.get('lr_adaptation', {})
            }

            # 打印完整的代理配置
            print("\n=== 创建DQN代理的完整配置 ===")
            print(f"基础学习率: {agent_config['learning_rate']}")
            print(f"最大学习率: {agent_config['max_learning_rate']}")
            print(f"学习率自适应配置: {agent_config['lr_adaptation']}")
            print("="*50)

            self.agent = DQNAgent(
                state_dim=state_size,
                action_dim=action_size,
                config=agent_config
            )
            
            # 检查是否有上一次训练的模型权重需要加载
            if self.load_model_path:
                try:
                    self.agent.load_weights(self.load_model_path)
                    print(f"DEBUG: 已加载预训练模型: {self.load_model_path}")
                except Exception as e:
                    print(f"警告: 无法加载模型权重: {e}")
            
            # 创建训练器
            from rl_strategies.trainer import RLTrainer
            
            # 创建完整的代理配置
            complete_agent_config = {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'update_target_every': update_target_every,
                'verbose': verbose,
                'max_learning_rate': self.train_config.get('max_learning_rate'),
                'lr_adaptation': self.train_config.get('lr_adaptation', {})
            }

            # 打印完整的代理配置
            print("\n=== 创建训练器时的完整代理配置 ===")
            print(f"基础学习率: {complete_agent_config['learning_rate']}")
            print(f"最大学习率: {complete_agent_config['max_learning_rate']}")
            print(f"学习率自适应配置: {complete_agent_config['lr_adaptation']}")
            print("="*50)
            
            self.trainer = RLTrainer(
                agent_type=agent_type,
                train_data=self.train_df,
                val_data=self.eval_df,
                env_config=env_config,
                agent_config=complete_agent_config
            )
            
            # 设置训练器的环境已正确设置环境类型
            self.trainer.train_env.env_type = 'training'
            self.trainer.val_env.env_type = 'evaluation'
            
            # 初始化交易记录存储
            self.trainer.train_env.train_transaction_history = []
            self.trainer.val_env.eval_transaction_history = []
            
            # 设置训练器的progress_callback
            # 使用内部处理方法，避免命名冲突
            self.trainer.progress_callback = lambda data: self.process_progress_data(data)
            
            # 设置训练状态和信号
            self.signal_training_started.emit()
            
            # 开始训练
            print("DEBUG: 开始训练，总回合数: {}".format(train_episodes))
            self.trainer.train(episodes=train_episodes, eval_interval=20, verbose=verbose)
            
            # 在训练结束后检查是否是因为停止请求而结束
            if self.stop_requested:
                self.log_signal.emit("训练已按请求停止")
                print("DEBUG: 训练已按请求停止")
            else:
                self.log_signal.emit("训练已完成所有轮次")
                print("DEBUG: 训练已完成所有轮次")
            
            # 训练完成后评估并保存模型
            if not self.stop_requested:
                # 获取最佳模型权重
                if hasattr(self.trainer, 'get_best_model_weights'):
                    best_weights = self.trainer.get_best_model_weights()
                else:
                    # 如果方法不存在，使用当前权重作为最佳权重
                    print("警告: trainer对象没有get_best_model_weights方法，使用当前权重")
                    best_weights = self.agent.get_weights() if hasattr(self.agent, 'get_weights') else None
                
                if best_weights is not None:
                    # 应用最佳权重并保存
                    self.agent.set_weights(best_weights)
                    
                    # 保存模型
                    if self.save_model_path:
                        self.trainer.save_model(self.save_model_path)
                        print(f"已保存最佳模型到: {self.save_model_path}")
                else:
                    print("WARNING: 未找到有效的最佳模型权重")
            
            # 发出训练完成信号
            self.signal_training_completed.emit()
        
        except Exception as e:
            import traceback
            error_msg = f"训练过程出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.signal_training_error.emit(error_msg)
        
        finally:
            self.running = False

    def evaluate_model(self, episode):
        """评估模型性能"""
        try:
            # 使用验证环境评估模型
            eval_results = self.trainer.evaluate(episodes=1)
            
            # 处理evaluate可能返回的元组值
            if isinstance(eval_results, tuple):
                if len(eval_results) == 2:
                    # 如果返回的是(success, results)格式的元组
                    success, eval_results = eval_results
                    if not success:
                        print(f"评估失败: {eval_results}")
                        return
                else:
                    print(f"警告: evaluate方法返回了意外的元组格式: {eval_results}")
                    return
            elif not isinstance(eval_results, dict):
                print(f"警告: evaluate方法返回了非字典结果: {type(eval_results)}")
                return
            
            # 记录评估结果
            print(f"评估结果 - 回合 {episode}:")
            print(f"平均奖励: {eval_results.get('avg_reward', 0):.4f}")
            print(f"平均收益率: {eval_results.get('avg_return', 0):.2f}%")
            print(f"平均交易次数: {eval_results.get('avg_trade_count', 0)}")
            print(f"胜率: {eval_results.get('win_rate', 0):.2f}%")
            
            # 回调处理评估结果
            self.eval_callback(eval_results)
                
        except Exception as e:
            print(f"评估模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 详细打印错误堆栈
            self.log_signal.emit(f"评估模型时出错: {str(e)}")
            # 继续训练，不中断