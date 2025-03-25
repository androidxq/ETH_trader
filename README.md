# ETH 量化交易系统

## 项目简介
这是一个基于遗传算法的 ETH 量化交易系统，使用符号回归方法挖掘交易因子。

## 主要功能
- 数据获取：自动下载历史K线数据
- 因子挖掘：使用遗传算法进行因子挖掘
- 参数优化：网格搜索优化算法参数
- 多进程支持：并行执行参数搜索
- 结果分析：自动生成分析报告
- 进度监控UI：可视化显示挖掘进度和内存占用
- 内存优化：自动管理进程内存，防止内存泄漏

## 项目结构
```
ETH_trader/
├── data/                  # 数据存储目录
│   └── kline/            # K线数据
├── data_fetcher/         # 数据获取模块
├── factor_research/      # 因子研究模块
│   ├── config/          # 配置文件
│   ├── data_loader.py   # 数据加载
│   └── symbolic_miner.py # 因子挖掘
├── scripts/              # 执行脚本
│   ├── download_history.py  # 下载历史数据
│   └── grid_search_factors.py # 网格搜索
├── tests/                # 测试用例
└── results/              # 结果输出
    └── grid_search/      # 网格搜索结果
```

## 配置系统
所有配置参数统一在 `factor_research/config/config.py` 中管理：
- 数据配置 (DATA_CONFIG)
- 遗传算法配置 (GA_BASE_CONFIG)
- 网格搜索配置 (GRID_SEARCH_CONFIG)
- 多进程配置 (MULTIPROCESSING_CONFIG)
- 进度显示配置 (PROGRESS_CONFIG)
- 日志配置 (LOGGING_CONFIG)

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载历史数据
```bash
python scripts/download_history.py
```

### 3. 运行网格搜索
```bash
python scripts/grid_search_factors.py
```

### 4. 启动UI监控界面
```bash
python scripts/run_grid_search_ui.py
```

## 参数说明
### 遗传算法参数
- population_size: 种群大小
- generations: 进化代数
- tournament_size: 锦标赛大小
- stopping_criteria: 停止条件阈值
- early_stopping: 早停代数
- const_range: 常数范围

### 网格搜索参数
- forward_period: 预测周期 [6, 12, 24]
- generations: 进化代数 [50, 100, 200]
- population_size: 种群大小 [2000, 4000, 8000]
- tournament_size: 锦标赛大小 [2, 5, 8]

## 结果分析
系统会自动生成以下结果：
1. CSV格式的详细结果
2. Markdown格式的分析报告
3. 最佳参数组合推荐

## 开发日志
详细开发记录请查看 [DEVLOG.md](DEVLOG.md)

## 决策记录
重要决策记录请查看 [DECISIONS.md](DECISIONS.md)

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
- v0.3.0 (2025-03-20 09:15:42)
  - 新增PyQtGraph高性能K线图显示模块
  - 支持历史K线数据浏览与交互
  - 优化数据加载和缓存机制
  - 实现鼠标滚轮缩放和键盘导航功能

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

### 4. UI监控模块 (grid_search_ui)
- 因子挖掘进度实时显示
- 进程内存占用监控
- 搜索结果动态查看
- K线数据可视化
- 因子性能评估（计划中）

### 5. 策略模块 (strategy)
- 因子组合策略
- 交易信号生成
- 仓位管理
- 风险控制

### 6. 回测模块 (backtest)
- 历史数据回测
- 策略评估
- 绩效分析

### 7. 交易执行模块 (execution)
- 交易所接口对接
- 订单管理
- 仓位跟踪

### 8. 监控模块 (monitor)
- 策略监控
- 资金监控
- 风险监控
- 异常报警

### 9. K线图显示模块 (kline_view)
- 基于PyQtGraph的高性能K线图
- 支持多币种、多时间周期切换
- 实时交互：缩放、拖动、跳转到日期
- 支持查看完整历史数据
- 键盘快捷键导航

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
│   ├── download_history.py # 历史数据下载 (已修复数据下载bug)
│   ├── grid_search_factors.py # 参数网格搜索
│   ├── kline_view_pyqtgraph.py # K线图显示模块
│   ├── run_kline_view_pyqtgraph.py # 运行K线图显示
├── results/                # 结果存储
│   ├── grid_search/        # 网格搜索结果
├── tests/                  # 单元测试
│   ├── test_data_fetch.py  # 数据获取测试
│   ├── test_factor_research.py # 因子研究测试
│   ├── test_symbolic_mining.py # 符号回归测试
├── DECISIONS.md            # 决策记录
├── DEVLOG.md               # 开发日志
└── README.md               # 项目说明
```

## 运行K线图显示
```bash
python scripts/run_kline_view_pyqtgraph.py
```

### K线图功能
- 支持多时间周期：1分钟至日线
- 币种：ETHUSDT永续合约
- 交互功能：
  - 鼠标滚轮：放大/缩小K线视图
  - 左右按键：导航历史数据
  - Home键：重置到最新数据
  - +/-键：调整视图范围
  - 跳转到日期：精确定位历史时间点
- 显示内容：
  - K线实体（红绿颜色区分涨跌）
  - 成交量柱状图
  - 十字光标位置信息
  - 价格和时间信息