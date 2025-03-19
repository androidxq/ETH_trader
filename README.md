# ETH 交易系统

## 项目概述
本项目是一个基于币安交易所的 ETH 自动交易系统，支持数据获取、因子生成、策略回测和实盘交易。

## 时间规范
- 所有文档中的时间戳必须使用北京时间（东八区，UTC+8）
- 时间格式统一为：YYYY-MM-DD HH:MM:SS
- 时间来源必须是互联网可靠时间源（如国家授时中心、北京时间在线服务）
- 所有开发记录需标注精确到秒的时间戳

## 版本历史
- v0.1.0 (2025-03-18 10:30:00)
  - 项目初始化
  - 完成基础架构设计
  - 实现数据获取模块
  - 开发符号回归因子挖掘器
- v0.2.0 (2025-03-19 15:45:20)
  - 实现参数网格搜索功能
  - 改进因子评估系统
  - 优化数据处理路径适配
- v0.2.1 (2025-03-19 16:20:35)
  - 修复符号回归因子挖掘器早停问题
  - 优化遗传算法参数配置
  - 改进适应度计算方法
  - 增强演化过程监控机制

## 系统架构
系统分为以下主要模块：

### 1. 数据获取模块 (data_fetcher)
- 币安 API 接入
- 实时行情和历史数据获取
- 支持秒级/分钟级数据

### 2. 数据存储模块 (data_storage)
- CSV 文件存储
- 按时间周期分文件管理
- 数据清洗和预处理

### 3. 因子研究模块 (factor_research)
- 技术指标计算
- 自定义因子生成
- 符号回归因子挖掘
- 因子评估和筛选
- 参数网格搜索优化

### 4. 策略模块 (strategy)
- 因子组合策略
- 交易信号生成
- 仓位管理
- 风险控制

### 5. 回测模块 (backtest)
- 历史数据回测
- 策略评估
- 绩效分析

### 6. 交易执行模块 (execution)
- 交易所接口对接
- 订单管理
- 仓位跟踪

### 7. 监控模块 (monitor)
- 策略监控
- 资金监控
- 风险监控
- 异常报警

## 项目结构
```plaintext
ETH_trader/
├── config/                 # 配置文件
│   ├── api_config.py       # API配置
│   ├── trading_config.py   # 交易配置
├── data_fetcher/           # 数据获取模块
│   ├── binance_fetcher.py  # 币安数据获取器
├── data_storage/           # 数据存储模块
│   ├── csv_storage.py      # CSV文件存储
├── factor_research/        # 因子研究模块
│   ├── base_factor.py      # 因子基类
│   ├── factor_evaluator.py # 因子评估器
│   ├── optimal_points.py   # 最优点标注工具
│   ├── symbolic_miner.py   # 符号回归因子挖掘器
├── scripts/                # 脚本工具
│   ├── download_history.py # 历史数据下载
│   ├── grid_search_factors.py # 参数网格搜索
├── results/                # 结果存储
│   ├── grid_search/        # 网格搜索结果
├── tests/                  # 单元测试
│   ├── test_data_fetch.py  # 数据获取测试
│   ├── test_factor_research.py # 因子研究测试
│   ├── test_symbolic_mining.py # 符号回归测试
├── DECISIONS.md            # 决策记录
├── DEVLOG.md               # 开发日志
└── README.md               # 项目说明