# ETH_trader 项目结构

## 项目概述
ETH_trader是一个基于Python的以太坊交易系统，集成了因子策略和强化学习策略两种交易方法。

## 目录结构

### 核心组件
- `crypto_trader.py`: 主程序入口文件
- `main_window.py`: 主窗口界面实现
- `factor_strategy_ui.py`: 因子策略UI界面
- `rl_strategies_ui.py`: 强化学习策略UI界面

### 配置目录
- `config/`
  - `strategy_params.json`: 策略参数配置
  - `trading_config.py`: 交易系统配置

### 数据处理
- `data_fetcher/`
  - `binance_fetcher.py`: 币安数据获取模块
- `data_storage/`
  - `csv_storage.py`: CSV数据存储实现
- `data/`
  - `kline/`: K线数据存储目录

### 因子研究
- `factor_research/`
  - `base_factor.py`: 基础因子类
  - `data_loader.py`: 数据加载器
  - `factor_evaluator.py`: 因子评估器
  - `optimal_points.py`: 最优点计算
  - `symbolic_miner.py`: 符号挖掘器
  - `config/`: 因子配置目录

### 强化学习策略
- `rl_strategies/`
  - `agents/`: 强化学习代理实现
  - `environments/`: 交易环境实现
  - `models/`: 神经网络模型
  - `utils/`: 工具函数
  - `config.py`: 强化学习配置
  - `trainer.py`: 训练器实现
  - `rl_training_thread.py`: 训练线程

### 模型和结果
- `saved_models/`: 保存的模型文件
- `results/`
  - `grid_search/`: 网格搜索结果
- `trading_results/`: 交易结果存储

### 示例和测试
- `examples/`
  - `support_resistance_strategy.py`: 支撑阻力策略示例
- `tests/`: 测试用例目录

### 文档
- `docs/`
  - `data_collection.md`: 数据收集文档
  - `factor_research.md`: 因子研究文档
  - `setup.md`: 环境搭建文档
  - `strategy_development.md`: 策略开发文档
  - `ui_issues.md`: UI问题记录

### 其他
- `scripts/`: 辅助脚本目录
- `requirements.txt`: 项目依赖
- `DECISIONS.md`: 决策记录
- `DEVLOG.md`: 开发日志
- `README.md`: 项目说明