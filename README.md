# ETH智能交易系统

基于因子挖掘的ETH加密货币交易系统。

## 主要功能

- 数据获取：从Binance获取历史K线数据
- 因子挖掘：使用遗传算法自动挖掘有效交易因子
- 回测系统：评估交易策略的历史表现
- 实时交易：接入Binance API进行自动化交易
- 多因子类型支持：包括量价获利因子、支撑阻力因子、趋势动能因子、波动率因子、流动性因子等

## 因子类型

系统当前支持以下几种因子类型：

1. **量价获利因子**：结合价格和交易量特征，挖掘市场获利机会。
2. **支撑阻力因子**：识别重要价格水平，捕捉价格反转和突破时机。特征包括：
   - 历史高低点位
   - 斐波那契回撤水平
   - 布林带作为动态支撑阻力
   - 价格突破模式
   - 价格形态识别
3. **趋势动能因子**：识别价格趋势的方向和强度，捕捉趋势行情。
4. **波动率因子**：基于价格波动特征，适合在剧烈波动环境中交易。
5. **流动性因子**：关注交易量和流动性变化，识别大资金进出市场的信号。

## 使用示例

### 挖掘支撑阻力因子并回测

```python
# 导入必要库
from crypto_trader import mine_new_factors, backtest_strategy
from data_processors.binance_processor import load_historical_data

# 加载数据
df = load_historical_data(symbol="ETHUSDT", interval="1h", start_date="2023-01-01")

# 挖掘支撑阻力因子
support_resistance_factors = mine_new_factors(
    data=df,
    factor_type="支撑阻力因子",  # 指定因子类型
    forward_period=24,  # 24小时预测期
    n_best=3  # 返回前3个最佳因子
)

# 查看挖掘结果
for i, factor in enumerate(support_resistance_factors):
    print(f"{i+1}. 表达式: {factor['expression']}")
    print(f"   IC: {factor['ic']:.4f}, Sharpe: {factor['sharpe']:.4f}")
```

更多详细示例请参考 `examples/` 目录下的示例代码。

## 安装

1. 克隆代码库:
```
git clone https://github.com/yourusername/eth_trader.git
cd eth_trader
```

2. 安装依赖:
```
pip install -r requirements.txt
```

## 配置

在 `config/` 目录下创建 `api_config.json` 文件，添加您的Binance API密钥：

```json
{
    "binance": {
        "api_key": "YOUR_API_KEY",
        "api_secret": "YOUR_API_SECRET"
    }
}
```

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
- v0.3.1 (2025-03-27 23:40:00)
  - 修复K线图数据加载和显示问题
  - 解决Windows平台上CSV文件读取兼容性问题
  - 优化视图交互处理，解决递归调用栈溢出问题
  - 改进负索引处理，支持无限制浏览历史数据
  - 优化时间戳转换和数据查找算法
- v0.3.2 (2025-03-28 10:30:00)
  - 修复网格搜索停止功能，解决进程无法终止问题
  - 实现多级进程终止策略，确保所有子进程正确停止
  - 添加超时和强制终止机制，防止进程卡死
  - 优化UI反馈，提供更清晰的停止过程状态信息
  - 增强异常处理，提高系统稳定性
- v0.3.3 (2025-03-28 14:45:20)
  - 修复网格搜索UI中的空值处理问题
  - 增强数据处理过程中的异常检查
  - 改进防御性编程实践，提高代码鲁棒性
  - 优化错误处理机制，减少程序崩溃可能性
- v0.3.4 (2025-03-28 22:30:15)
  - 重构K线图交易标记渲染机制，显著提升性能
  - 修复回测后K线图卡顿问题，优化交互体验
  - 实现增量渲染和视图变化检测，减少不必要的重绘
  - 添加调试模式控制，优化日志输出和资源使用
  - 新增交易标记可见性保证功能，确保标记在视图中可见
- v0.3.5 (2025-03-28 23:15:00)
  - 全面优化交易系统核心组件
  - 实现固定交易金额机制（100）
  - 增强风险控制（15%最大回撤限制）
  - 优化DQN网络结构（4层）
  - 改进奖励系统和交易频率控制
  - 修正交易费率为0.05%
  - 提升系统整体稳定性和性能

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

## 2025-03-28
### 19. 网格搜索停止机制重构 (2025-03-28 10:30:15)
- 决策：全面重构网格搜索停止机制
- 问题：
  - 网格搜索启动后无法有效停止
  - 点击停止按钮后进程继续运行
  - 无法释放系统资源
  - 用户不得不强制关闭程序

- 具体方案：
  1. 多级进程终止策略：
     - 记录并跟踪所有创建的子进程
     - 实现从父进程到子进程的终止传递
     - 添加进程组终止支持
     - 实现循环检查确保所有进程终止

  2. 超时与强制终止机制：
     - 实现10秒超时检测
     - 超时后使用强制终止（SIGKILL）
     - 添加进程存在性检查逻辑
     - 实现资源清理确保系统恢复

  3. 增强用户反馈：
     - 提供分阶段停止状态反馈
     - 显示正在终止的进程数量
     - 在异常情况下提供详细诊断信息
     - 明确指示强制终止的发生

- 技术决策要点：
  1. 使用psutil库管理进程，替代简单的terminate()调用
  2. 将进程状态检查频率从原来的每批次提高到每次迭代
  3. 将UI反馈模式从简单状态文本升级为详细的分阶段报告
  4. 添加日志记录功能记录停止过程中的关键事件

- 原因：
  - 原有停止机制过于简单，不能应对复杂的多进程环境
  - 缺乏超时机制导致进程卡住时无法恢复
  - 用户体验差，无法了解停止过程的进展
  - 资源无法释放影响系统稳定性和后续操作

- 影响：
  - 提高系统稳定性和可靠性
  - 改善用户体验，提供清晰的操作反馈
  - 避免资源泄漏，保护系统长期运行稳定性
  - 为未来添加更精细的任务控制奠定基础