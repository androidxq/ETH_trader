# 因子网格搜索报告

生成时间: 2025-03-28 10:09:52

## 总体统计

- 参数组合总数: 21
- 成功完成组合: 21
- 失败组合: 0

## 最佳因子 (按IC排序)

### 因子 1

- 表达式: `div(abs(X52), mul(X48, X22))`
- 预测能力(IC): 0.0644
- 稳定性: 0.7521
- 多头收益: 0.0019
- 空头收益: -0.0002
- 复杂度: 6
- 参数组合:
  - forward_period: 24
  - population_size: 1000
  - generations: 200
  - tournament_size: 30
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 2

- 表达式: `sub(sin(X46), div(X49, -0.077))`
- 预测能力(IC): 0.0376
- 稳定性: 0.9996
- 多头收益: 0.0011
- 空头收益: -0.0001
- 复杂度: 6
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 200
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 3

- 表达式: `mul(log(abs(X58)), div(div(X29, X18), cos(X47)))`
- 预测能力(IC): 0.0313
- 稳定性: 0.6469
- 多头收益: -0.0000
- 空头收益: -0.0003
- 复杂度: 10
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 300
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 4

- 表达式: `div(sin(cos(X20)), mul(mul(X30, X44), add(X48, X23)))`
- 预测能力(IC): 0.0301
- 稳定性: 0.6938
- 多头收益: 0.0008
- 空头收益: 0.0003
- 复杂度: 11
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 300
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 5

- 表达式: `div(sqrt(sub(sub(X16, X11), sqrt(X43))), sub(add(mul(X45, X21), neg(X42)), sin(mul(X53, X49))))`
- 预测能力(IC): 0.0287
- 稳定性: 0.1771
- 多头收益: 0.0025
- 空头收益: 0.0008
- 复杂度: 19
- 参数组合:
  - forward_period: 48
  - population_size: 1000
  - generations: 300
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 6

- 表达式: `abs(mul(div(log(X0), sin(X41)), mul(div(X18, X36), sqrt(X47))))`
- 预测能力(IC): 0.0228
- 稳定性: 0.1664
- 多头收益: -0.0000
- 空头收益: -0.0005
- 复杂度: 13
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 100
  - tournament_size: 30
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 7

- 表达式: `sqrt(div(mul(sub(X55, X20), div(X48, X12)), log(cos(X14))))`
- 预测能力(IC): 0.0223
- 稳定性: 0.5826
- 多头收益: 0.0003
- 空头收益: 0.0004
- 复杂度: 12
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 100
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 8

- 表达式: `mul(add(X12, X31), mul(X30, X21))`
- 预测能力(IC): 0.0212
- 稳定性: 0.9329
- 多头收益: 0.0001
- 空头收益: -0.0001
- 复杂度: 7
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 300
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 9

- 表达式: `div(log(neg(X51)), mul(abs(X50), sqrt(X3)))`
- 预测能力(IC): 0.0190
- 稳定性: 0.7241
- 多头收益: 0.0007
- 空头收益: 0.0002
- 复杂度: 9
- 参数组合:
  - forward_period: 36
  - population_size: 1000
  - generations: 200
  - tournament_size: 5
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

### 因子 10

- 表达式: `sub(cos(add(neg(cos(X9)), sub(sub(X61, X56), div(X22, X19)))), neg(abs(sub(add(X41, X52), div(X9, X24)))))`
- 预测能力(IC): 0.0190
- 稳定性: 0.1639
- 多头收益: 0.0027
- 空头收益: 0.0011
- 复杂度: 22
- 参数组合:
  - forward_period: 36
  - population_size: 1000
  - generations: 200
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.2
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 25

## 各参数组合结果

### 组合 1

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 100
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 77.75秒

#### 找到的因子:

##### 因子 1:

- 表达式: `sqrt(div(mul(sub(X55, X20), div(X48, X12)), log(cos(X14))))`
- 预测能力(IC): 0.0223
- 稳定性: 0.5826
- 多头收益: 0.0003
- 空头收益: 0.0004
- 复杂度: 12

##### 因子 2:

- 表达式: `div(sqrt(0.480), mul(X58, X15))`
- 预测能力(IC): -0.0208
- 稳定性: 0.0603
- 多头收益: 0.0002
- 空头收益: -0.0003
- 复杂度: 6

##### 因子 3:

- 表达式: `div(abs(X31), sub(X17, X25))`
- 预测能力(IC): -0.0097
- 稳定性: 0.0579
- 多头收益: -0.0000
- 空头收益: -0.0004
- 复杂度: 6

##### 因子 4:

- 表达式: `div(X6, X24)`
- 预测能力(IC): -0.0100
- 稳定性: -0.0257
- 多头收益: -0.0002
- 空头收益: 0.0001
- 复杂度: 3

##### 因子 5:

- 表达式: `add(sub(X49, X6), div(X18, X58))`
- 预测能力(IC): -0.0029
- 稳定性: 0.0819
- 多头收益: 0.0002
- 空头收益: -0.0003
- 复杂度: 7

### 组合 2

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 100
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 78.57秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X53, X14)`
- 预测能力(IC): -0.0165
- 稳定性: 0.6910
- 多头收益: 0.0002
- 空头收益: -0.0006
- 复杂度: 3

##### 因子 2:

- 表达式: `div(add(mul(X17, X46), mul(X49, X27)), add(sin(X15), neg(X48)))`
- 预测能力(IC): -0.0191
- 稳定性: 0.5485
- 多头收益: -0.0001
- 空头收益: -0.0009
- 复杂度: 13

##### 因子 3:

- 表达式: `div(log(X8), sin(X43))`
- 预测能力(IC): 0.0004
- 稳定性: 0.6967
- 多头收益: 0.0006
- 空头收益: 0.0005
- 复杂度: 5

##### 因子 4:

- 表达式: `abs(mul(div(log(X0), sin(X41)), mul(div(X18, X36), sqrt(X47))))`
- 预测能力(IC): 0.0228
- 稳定性: 0.1664
- 多头收益: -0.0000
- 空头收益: -0.0005
- 复杂度: 13

##### 因子 5:

- 表达式: `div(div(X42, X9), neg(X42))`
- 预测能力(IC): 0.0058
- 稳定性: -0.0048
- 多头收益: 0.0007
- 空头收益: 0.0000
- 复杂度: 6

### 组合 3

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 100
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 79.40秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(mul(X58, X21), mul(X47, X27))`
- 预测能力(IC): -0.0370
- 稳定性: 0.5594
- 多头收益: -0.0003
- 空头收益: -0.0003
- 复杂度: 7

##### 因子 2:

- 表达式: `div(log(X6), X23)`
- 预测能力(IC): 0.0097
- 稳定性: 0.1265
- 多头收益: 0.0003
- 空头收益: 0.0003
- 复杂度: 4

##### 因子 3:

- 表达式: `div(X34, X6)`
- 预测能力(IC): -0.0070
- 稳定性: 0.0232
- 多头收益: 0.0007
- 空头收益: -0.0004
- 复杂度: 3

##### 因子 4:

- 表达式: `abs(abs(div(div(sin(X12), add(X18, X40)), mul(mul(X43, X9), div(X44, X37)))))`
- 预测能力(IC): -0.0023
- 稳定性: 0.3991
- 多头收益: 0.0003
- 空头收益: -0.0002
- 复杂度: 16

##### 因子 5:

- 表达式: `sub(div(add(X10, X3), mul(X42, X27)), sin(sin(X59)))`
- 预测能力(IC): -0.0000
- 稳定性: 0.8553
- 多头收益: -0.0003
- 空头收益: -0.0003
- 复杂度: 11

### 组合 4

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 100
- tournament_size: 5
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 77.83秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X33, X40)`
- 预测能力(IC): -0.0065
- 稳定性: 0.2151
- 多头收益: -0.0001
- 空头收益: -0.0007
- 复杂度: 3

##### 因子 2:

- 表达式: `div(X34, X60)`
- 预测能力(IC): -0.0149
- 稳定性: 0.0886
- 多头收益: 0.0004
- 空头收益: -0.0004
- 复杂度: 3

##### 因子 3:

- 表达式: `div(sub(X58, X25), sub(X36, X15))`
- 预测能力(IC): -0.0026
- 稳定性: 0.0962
- 多头收益: 0.0006
- 空头收益: -0.0005
- 复杂度: 7

##### 因子 4:

- 表达式: `div(div(X40, X13), X9)`
- 预测能力(IC): 0.0168
- 稳定性: -0.0011
- 多头收益: 0.0004
- 空头收益: -0.0007
- 复杂度: 5

##### 因子 5:

- 表达式: `div(mul(sub(sub(add(X40, X23), sin(X51)), neg(log(X3))), div(div(sub(X36, X33), div(X25, X61)), add(sin(X50), div(X29, X25)))), cos(neg(div(div(X6, X6), div(X45, X25)))))`
- 预测能力(IC): 0.0012
- 稳定性: -0.0543
- 多头收益: 0.0006
- 空头收益: -0.0008
- 复杂度: 35

### 组合 5

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 200
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 159.73秒

#### 找到的因子:

##### 因子 1:

- 表达式: `sub(sin(X46), div(X49, -0.077))`
- 预测能力(IC): 0.0376
- 稳定性: 0.9996
- 多头收益: 0.0011
- 空头收益: -0.0001
- 复杂度: 6

##### 因子 2:

- 表达式: `add(div(sub(X33, X26), neg(X32)), sin(mul(X58, X10)))`
- 预测能力(IC): 0.0121
- 稳定性: 0.4151
- 多头收益: 0.0009
- 空头收益: 0.0001
- 复杂度: 11

##### 因子 3:

- 表达式: `div(X19, X20)`
- 预测能力(IC): 0.0091
- 稳定性: -0.0098
- 多头收益: 0.0004
- 空头收益: 0.0001
- 复杂度: 3

##### 因子 4:

- 表达式: `add(add(div(sub(sub(X19, X35), div(X0, X5)), sub(add(X20, X2), sin(X53))), cos(sin(div(X38, X50)))), neg(sin(cos(sub(X32, X29)))))`
- 预测能力(IC): 0.0129
- 稳定性: -0.0452
- 多头收益: 0.0007
- 空头收益: -0.0003
- 复杂度: 27

##### 因子 5:

- 表达式: `div(abs(neg(sub(div(abs(X6), sub(X56, X5)), sin(sqrt(X7))))), abs(neg(add(cos(sub(X16, X49)), sub(div(X12, X26), sqrt(X5))))))`
- 预测能力(IC): 0.0015
- 稳定性: 0.0222
- 多头收益: -0.0000
- 空头收益: -0.0000
- 复杂度: 26

### 组合 6

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 300
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 252.79秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(sin(cos(X20)), mul(mul(X30, X44), add(X48, X23)))`
- 预测能力(IC): 0.0301
- 稳定性: 0.6938
- 多头收益: 0.0008
- 空头收益: 0.0003
- 复杂度: 11

##### 因子 2:

- 表达式: `mul(add(X12, X31), mul(X30, X21))`
- 预测能力(IC): 0.0212
- 稳定性: 0.9329
- 多头收益: 0.0001
- 空头收益: -0.0001
- 复杂度: 7

##### 因子 3:

- 表达式: `mul(log(abs(X58)), div(div(X29, X18), cos(X47)))`
- 预测能力(IC): 0.0313
- 稳定性: 0.6469
- 多头收益: -0.0000
- 空头收益: -0.0003
- 复杂度: 10

##### 因子 4:

- 表达式: `div(X6, X32)`
- 预测能力(IC): -0.0122
- 稳定性: -0.0638
- 多头收益: -0.0004
- 空头收益: -0.0000
- 复杂度: 3

##### 因子 5:

- 表达式: `div(cos(neg(X45)), sin(neg(X33)))`
- 预测能力(IC): 0.0025
- 稳定性: 0.0277
- 多头收益: -0.0002
- 空头收益: -0.0003
- 复杂度: 7

### 组合 7

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 300
- tournament_size: 5
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 253.34秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X38, mul(X39, X34))`
- 预测能力(IC): -0.0097
- 稳定性: 0.6739
- 多头收益: 0.0001
- 空头收益: 0.0004
- 复杂度: 5

##### 因子 2:

- 表达式: `div(mul(abs(abs(X19)), add(mul(X24, X27), cos(X49))), neg(log(sub(X5, X8))))`
- 预测能力(IC): 0.0080
- 稳定性: 0.5052
- 多头收益: 0.0005
- 空头收益: -0.0003
- 复杂度: 16

##### 因子 3:

- 表达式: `div(X35, X60)`
- 预测能力(IC): -0.0126
- 稳定性: 0.0798
- 多头收益: 0.0005
- 空头收益: -0.0004
- 复杂度: 3

##### 因子 4:

- 表达式: `div(add(sin(sin(log(X61))), add(neg(sqrt(X14)), abs(abs(X48)))), log(cos(abs(sqrt(X33)))))`
- 预测能力(IC): -0.0082
- 稳定性: 0.2695
- 多头收益: -0.0006
- 空头收益: -0.0003
- 复杂度: 18

##### 因子 5:

- 表达式: `mul(neg(add(neg(add(X39, X33)), sin(div(X38, X1)))), neg(div(mul(sub(X57, X11), add(X56, X1)), sub(abs(X2), mul(X7, X52)))))`
- 预测能力(IC): -0.0299
- 稳定性: -0.0177
- 多头收益: -0.0003
- 空头收益: -0.0005
- 复杂度: 26

### 组合 8

#### 参数:

- forward_period: 24
- population_size: 1000
- generations: 100
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 78.16秒

#### 找到的因子:

##### 因子 1:

- 表达式: `mul(add(cos(sub(X10, X8)), sqrt(div(X10, X54))), add(sub(div(X50, X47), sqrt(X20)), sub(div(X18, X15), cos(X43))))`
- 预测能力(IC): -0.0412
- 稳定性: 0.6877
- 多头收益: 0.0011
- 空头收益: 0.0005
- 复杂度: 23

##### 因子 2:

- 表达式: `neg(div(X27, div(X54, X34)))`
- 预测能力(IC): -0.0144
- 稳定性: 0.9883
- 多头收益: -0.0013
- 空头收益: -0.0022
- 复杂度: 6

##### 因子 3:

- 表达式: `sub(div(neg(X28), mul(X53, X48)), sqrt(cos(X41)))`
- 预测能力(IC): -0.0178
- 稳定性: 0.6923
- 多头收益: -0.0003
- 空头收益: -0.0000
- 复杂度: 10

##### 因子 4:

- 表达式: `sub(X10, sub(div(mul(div(X11, X51), neg(X25)), log(sin(X2))), div(abs(neg(X46)), abs(add(X34, X40)))))`
- 预测能力(IC): -0.0248
- 稳定性: 0.5732
- 多头收益: 0.0009
- 空头收益: -0.0014
- 复杂度: 21

##### 因子 5:

- 表达式: `div(add(X10, X51), mul(X9, X11))`
- 预测能力(IC): 0.0001
- 稳定性: -0.0073
- 多头收益: 0.0001
- 空头收益: -0.0005
- 复杂度: 7

### 组合 9

#### 参数:

- forward_period: 24
- population_size: 1000
- generations: 100
- tournament_size: 5
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 77.00秒

#### 找到的因子:

##### 因子 1:

- 表达式: `add(add(abs(X30), add(X27, X8)), abs(div(X47, X38)))`
- 预测能力(IC): -0.0655
- 稳定性: 0.5514
- 多头收益: -0.0008
- 空头收益: -0.0021
- 复杂度: 11

##### 因子 2:

- 表达式: `div(X41, X11)`
- 预测能力(IC): -0.0127
- 稳定性: 0.4598
- 多头收益: 0.0001
- 空头收益: -0.0016
- 复杂度: 3

##### 因子 3:

- 表达式: `div(sin(X8), sin(X21))`
- 预测能力(IC): -0.0012
- 稳定性: 0.0962
- 多头收益: 0.0015
- 空头收益: -0.0003
- 复杂度: 5

##### 因子 4:

- 表达式: `add(sqrt(sin(add(X40, 0.752))), abs(div(neg(X11), div(X5, X8))))`
- 预测能力(IC): 0.0186
- 稳定性: 0.0517
- 多头收益: 0.0003
- 空头收益: -0.0003
- 复杂度: 13

##### 因子 5:

- 表达式: `add(sin(cos(log(X9))), div(sub(add(X47, X57), abs(X13)), mul(mul(X5, X56), add(X1, X14))))`
- 预测能力(IC): -0.0070
- 稳定性: 0.0066
- 多头收益: 0.0006
- 空头收益: -0.0003
- 复杂度: 19

### 组合 10

#### 参数:

- forward_period: 24
- population_size: 1000
- generations: 200
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 157.88秒

#### 找到的因子:

##### 因子 1:

- 表达式: `sub(add(div(X39, X50), sqrt(X9)), neg(sqrt(X15)))`
- 预测能力(IC): -0.0058
- 稳定性: 0.9643
- 多头收益: 0.0003
- 空头收益: 0.0007
- 复杂度: 10

##### 因子 2:

- 表达式: `neg(div(sin(log(sin(X32))), log(cos(log(X27)))))`
- 预测能力(IC): -0.0043
- 稳定性: 0.7920
- 多头收益: 0.0008
- 空头收益: -0.0004
- 复杂度: 10

##### 因子 3:

- 表达式: `abs(sub(div(sub(log(sub(X52, X8)), sub(add(X13, X2), mul(X58, X6))), div(div(abs(X45), abs(X2)), neg(log(X14)))), sin(cos(log(sub(X61, X25))))))`
- 预测能力(IC): -0.0214
- 稳定性: 0.4444
- 多头收益: 0.0004
- 空头收益: -0.0010
- 复杂度: 30

##### 因子 4:

- 表达式: `sub(sin(sin(X44)), div(div(X4, X59), add(X21, X23)))`
- 预测能力(IC): 0.0073
- 稳定性: -0.0332
- 多头收益: 0.0006
- 空头收益: -0.0000
- 复杂度: 11

##### 因子 5:

- 表达式: `div(sub(sin(sub(X3, X60)), div(div(X42, X10), abs(X35))), sin(sin(sub(X52, X21))))`
- 预测能力(IC): 0.0121
- 稳定性: -0.0246
- 多头收益: -0.0003
- 空头收益: -0.0006
- 复杂度: 17

### 组合 11

#### 参数:

- forward_period: 24
- population_size: 1000
- generations: 200
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 160.33秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(abs(X52), mul(X48, X22))`
- 预测能力(IC): 0.0644
- 稳定性: 0.7521
- 多头收益: 0.0019
- 空头收益: -0.0002
- 复杂度: 6

##### 因子 2:

- 表达式: `div(X41, X50)`
- 预测能力(IC): 0.0116
- 稳定性: 0.9785
- 多头收益: 0.0011
- 空头收益: -0.0009
- 复杂度: 3

##### 因子 3:

- 表达式: `div(cos(X0), sub(X43, X19))`
- 预测能力(IC): -0.0052
- 稳定性: 0.6195
- 多头收益: -0.0006
- 空头收益: -0.0003
- 复杂度: 6

##### 因子 4:

- 表达式: `sub(add(div(sin(sub(neg(X39), log(X0))), div(div(mul(X56, X57), sub(X52, X31)), mul(cos(X49), div(X38, X6)))), abs(cos(neg(mul(X8, X61))))), log(abs(log(sqrt(add(X19, X34))))))`
- 预测能力(IC): 0.0065
- 稳定性: 0.0029
- 多头收益: 0.0007
- 空头收益: -0.0005
- 复杂度: 36

##### 因子 5:

- 表达式: `sub(div(sub(cos(X31), add(-0.707, X12)), sin(sin(X9))), mul(add(mul(X3, X26), sqrt(X12)), mul(cos(X24), log(X11))))`
- 预测能力(IC): 0.0091
- 稳定性: -0.0031
- 多头收益: 0.0003
- 空头收益: -0.0006
- 复杂度: 23

### 组合 12

#### 参数:

- forward_period: 24
- population_size: 1000
- generations: 200
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 161.19秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(sub(X39, X2), neg(X31))`
- 预测能力(IC): 0.0179
- 稳定性: 0.8727
- 多头收益: 0.0008
- 空头收益: 0.0006
- 复杂度: 6

##### 因子 2:

- 表达式: `div(X28, X48)`
- 预测能力(IC): -0.0040
- 稳定性: 0.4059
- 多头收益: 0.0007
- 空头收益: -0.0003
- 复杂度: 3

##### 因子 3:

- 表达式: `div(X58, X16)`
- 预测能力(IC): -0.0115
- 稳定性: 0.1472
- 多头收益: 0.0006
- 空头收益: -0.0007
- 复杂度: 3

##### 因子 4:

- 表达式: `div(X21, X59)`
- 预测能力(IC): 0.0124
- 稳定性: 0.0223
- 多头收益: -0.0002
- 空头收益: -0.0004
- 复杂度: 3

##### 因子 5:

- 表达式: `div(log(X34), log(X39))`
- 预测能力(IC): 0.0002
- 稳定性: 0.8875
- 多头收益: -0.0004
- 空头收益: -0.0006
- 复杂度: 5

### 组合 13

#### 参数:

- forward_period: 36
- population_size: 1000
- generations: 200
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 157.87秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(neg(div(X49, X1)), mul(cos(X22), sqrt(X12)))`
- 预测能力(IC): -0.0037
- 稳定性: 0.4189
- 多头收益: 0.0009
- 空头收益: 0.0000
- 复杂度: 10

##### 因子 2:

- 表达式: `abs(div(log(mul(X43, X16)), mul(mul(X27, X57), log(X1))))`
- 预测能力(IC): -0.0302
- 稳定性: 0.3690
- 多头收益: -0.0003
- 空头收益: -0.0020
- 复杂度: 12

##### 因子 3:

- 表达式: `add(div(X56, X47), div(X9, X28))`
- 预测能力(IC): 0.0024
- 稳定性: 0.2632
- 多头收益: 0.0006
- 空头收益: -0.0002
- 复杂度: 7

##### 因子 4:

- 表达式: `div(add(add(sin(X19), sub(X31, X39)), sqrt(abs(X2))), sin(mul(abs(X9), abs(X8))))`
- 预测能力(IC): -0.0100
- 稳定性: 0.1597
- 多头收益: -0.0000
- 空头收益: -0.0020
- 复杂度: 17

##### 因子 5:

- 表达式: `div(sub(sub(X24, X18), sub(X1, X36)), sin(mul(X41, X22)))`
- 预测能力(IC): -0.0001
- 稳定性: 0.4453
- 多头收益: 0.0013
- 空头收益: -0.0002
- 复杂度: 12

### 组合 14

#### 参数:

- forward_period: 36
- population_size: 1000
- generations: 200
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 159.94秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(add(div(X50, X35), log(X9)), div(div(X18, X41), cos(X53)))`
- 预测能力(IC): 0.0084
- 稳定性: 0.6728
- 多头收益: 0.0013
- 空头收益: -0.0011
- 复杂度: 13

##### 因子 2:

- 表达式: `div(sqrt(sub(div(X60, X19), neg(X19))), abs(sub(add(X52, X19), add(X1, X54))))`
- 预测能力(IC): -0.0275
- 稳定性: 0.3893
- 多头收益: -0.0013
- 空头收益: -0.0034
- 复杂度: 16

##### 因子 3:

- 表达式: `div(add(X5, X3), mul(X51, X56))`
- 预测能力(IC): -0.0101
- 稳定性: 0.1595
- 多头收益: 0.0005
- 空头收益: -0.0003
- 复杂度: 7

##### 因子 4:

- 表达式: `sub(sqrt(add(X22, X55)), div(log(X59), neg(X33)))`
- 预测能力(IC): -0.0147
- 稳定性: 0.1963
- 多头收益: -0.0005
- 空头收益: 0.0004
- 复杂度: 10

##### 因子 5:

- 表达式: `div(cos(X40), X6)`
- 预测能力(IC): 0.0019
- 稳定性: -0.0048
- 多头收益: 0.0010
- 空头收益: -0.0007
- 复杂度: 4

### 组合 15

#### 参数:

- forward_period: 36
- population_size: 1000
- generations: 200
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 160.82秒

#### 找到的因子:

##### 因子 1:

- 表达式: `sub(cos(add(neg(cos(X9)), sub(sub(X61, X56), div(X22, X19)))), neg(abs(sub(add(X41, X52), div(X9, X24)))))`
- 预测能力(IC): 0.0190
- 稳定性: 0.1639
- 多头收益: 0.0027
- 空头收益: 0.0011
- 复杂度: 22

##### 因子 2:

- 表达式: `div(sin(X46), X50)`
- 预测能力(IC): 0.0028
- 稳定性: 0.9803
- 多头收益: 0.0000
- 空头收益: 0.0007
- 复杂度: 4

##### 因子 3:

- 表达式: `neg(div(log(X59), neg(X33)))`
- 预测能力(IC): -0.0162
- 稳定性: 0.1964
- 多头收益: -0.0007
- 空头收益: -0.0003
- 复杂度: 6

##### 因子 4:

- 表达式: `add(sub(sqrt(log(X27)), sin(log(X44))), sub(div(mul(X23, X53), mul(X36, X61)), sub(mul(X42, X38), log(X11))))`
- 预测能力(IC): 0.0114
- 稳定性: 0.1061
- 多头收益: 0.0010
- 空头收益: -0.0012
- 复杂度: 22

##### 因子 5:

- 表达式: `div(neg(log(X57)), sub(mul(X23, X0), abs(X52)))`
- 预测能力(IC): -0.0027
- 稳定性: 0.0374
- 多头收益: 0.0005
- 空头收益: -0.0013
- 复杂度: 10

### 组合 16

#### 参数:

- forward_period: 36
- population_size: 1000
- generations: 200
- tournament_size: 5
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 157.68秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(log(neg(X51)), mul(abs(X50), sqrt(X3)))`
- 预测能力(IC): 0.0190
- 稳定性: 0.7241
- 多头收益: 0.0007
- 空头收益: 0.0002
- 复杂度: 9

##### 因子 2:

- 表达式: `div(div(X25, X40), X27)`
- 预测能力(IC): 0.0037
- 稳定性: 0.4597
- 多头收益: 0.0023
- 空头收益: 0.0002
- 复杂度: 5

##### 因子 3:

- 表达式: `div(div(X48, X3), neg(X3))`
- 预测能力(IC): 0.0180
- 稳定性: 0.0391
- 多头收益: 0.0010
- 空头收益: 0.0018
- 复杂度: 6

##### 因子 4:

- 表达式: `add(div(add(X44, X38), neg(X40)), abs(neg(X15)))`
- 预测能力(IC): 0.0103
- 稳定性: 0.2425
- 多头收益: 0.0011
- 空头收益: -0.0005
- 复杂度: 10

##### 因子 5:

- 表达式: `div(add(X19, X33), X6)`
- 预测能力(IC): -0.0028
- 稳定性: 0.0174
- 多头收益: 0.0019
- 空头收益: -0.0014
- 复杂度: 5

### 组合 17

#### 参数:

- forward_period: 48
- population_size: 1000
- generations: 200
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 160.63秒

#### 找到的因子:

##### 因子 1:

- 表达式: `add(sin(sqrt(X42)), div(cos(X19), sin(X26)))`
- 预测能力(IC): -0.0243
- 稳定性: 0.5278
- 多头收益: -0.0017
- 空头收益: -0.0009
- 复杂度: 9

##### 因子 2:

- 表达式: `div(X31, X12)`
- 预测能力(IC): -0.0139
- 稳定性: 0.0350
- 多头收益: 0.0015
- 空头收益: -0.0007
- 复杂度: 3

##### 因子 3:

- 表达式: `sub(log(X44), div(X3, X61))`
- 预测能力(IC): -0.0026
- 稳定性: 0.0556
- 多头收益: 0.0019
- 空头收益: -0.0010
- 复杂度: 6

##### 因子 4:

- 表达式: `div(mul(mul(cos(X23), log(X48)), sub(cos(X46), add(X12, X16))), div(neg(add(X0, X43)), abs(abs(X60))))`
- 预测能力(IC): -0.0083
- 稳定性: -0.0305
- 多头收益: -0.0007
- 空头收益: -0.0007
- 复杂度: 21

##### 因子 5:

- 表达式: `sub(X0, neg(div(log(X0), neg(X53))))`
- 预测能力(IC): 0.0000
- 稳定性: 0.4654
- 多头收益: -0.0006
- 空头收益: 0.0006
- 复杂度: 8

### 组合 18

#### 参数:

- forward_period: 48
- population_size: 1000
- generations: 200
- tournament_size: 5
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 158.91秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X0, X13)`
- 预测能力(IC): -0.0056
- 稳定性: 0.1098
- 多头收益: 0.0025
- 空头收益: -0.0000
- 复杂度: 3

##### 因子 2:

- 表达式: `div(X16, X59)`
- 预测能力(IC): -0.0144
- 稳定性: 0.0892
- 多头收益: 0.0005
- 空头收益: -0.0012
- 复杂度: 3

##### 因子 3:

- 表达式: `div(X60, X12)`
- 预测能力(IC): -0.0170
- 稳定性: 0.0678
- 多头收益: 0.0004
- 空头收益: -0.0011
- 复杂度: 3

##### 因子 4:

- 表达式: `div(X17, X57)`
- 预测能力(IC): -0.0085
- 稳定性: 0.0126
- 多头收益: 0.0002
- 空头收益: -0.0011
- 复杂度: 3

##### 因子 5:

- 表达式: `div(X33, X59)`
- 预测能力(IC): 0.0016
- 稳定性: 0.0166
- 多头收益: 0.0007
- 空头收益: -0.0018
- 复杂度: 3

### 组合 19

#### 参数:

- forward_period: 48
- population_size: 1000
- generations: 300
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 243.86秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(sqrt(sub(sub(X16, X11), sqrt(X43))), sub(add(mul(X45, X21), neg(X42)), sin(mul(X53, X49))))`
- 预测能力(IC): 0.0287
- 稳定性: 0.1771
- 多头收益: 0.0025
- 空头收益: 0.0008
- 复杂度: 19

##### 因子 2:

- 表达式: `div(X28, X51)`
- 预测能力(IC): -0.0374
- 稳定性: 0.8921
- 多头收益: -0.0002
- 空头收益: -0.0014
- 复杂度: 3

##### 因子 3:

- 表达式: `div(X17, X43)`
- 预测能力(IC): -0.0134
- 稳定性: 0.6085
- 多头收益: -0.0002
- 空头收益: -0.0021
- 复杂度: 3

##### 因子 4:

- 表达式: `add(div(div(X57, X36), neg(X15)), sin(sin(X52)))`
- 预测能力(IC): 0.0110
- 稳定性: 0.0134
- 多头收益: 0.0018
- 空头收益: -0.0004
- 复杂度: 10

##### 因子 5:

- 表达式: `div(X40, X6)`
- 预测能力(IC): -0.0084
- 稳定性: 0.0388
- 多头收益: 0.0012
- 空头收益: -0.0015
- 复杂度: 3

### 组合 20

#### 参数:

- forward_period: 48
- population_size: 3000
- generations: 100
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 244.82秒

#### 找到的因子:

##### 因子 1:

- 表达式: `neg(div(sqrt(X3), mul(X55, X32)))`
- 预测能力(IC): -0.0360
- 稳定性: 0.7732
- 多头收益: -0.0008
- 空头收益: -0.0022
- 复杂度: 7

##### 因子 2:

- 表达式: `div(div(-0.777, X28), abs(X52))`
- 预测能力(IC): 0.0048
- 稳定性: 0.0250
- 多头收益: 0.0020
- 空头收益: 0.0017
- 复杂度: 6

##### 因子 3:

- 表达式: `div(sqrt(X59), mul(X60, X8))`
- 预测能力(IC): -0.0026
- 稳定性: 0.1840
- 多头收益: 0.0009
- 空头收益: -0.0003
- 复杂度: 6

##### 因子 4:

- 表达式: `div(cos(neg(abs(X12))), mul(add(add(X31, X42), cos(X17)), abs(div(X52, X17))))`
- 预测能力(IC): -0.0100
- 稳定性: 0.1836
- 多头收益: -0.0003
- 空头收益: -0.0029
- 复杂度: 16

##### 因子 5:

- 表达式: `sub(add(div(log(X47), mul(X21, X34)), sub(log(X26), sqrt(X40))), sqrt(abs(neg(X44))))`
- 预测能力(IC): -0.0019
- 稳定性: 0.4062
- 多头收益: -0.0000
- 空头收益: -0.0004
- 复杂度: 17

### 组合 21

#### 参数:

- forward_period: 48
- population_size: 3000
- generations: 100
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.2
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 25

#### 运行时间: 235.04秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(div(X26, X49), mul(X16, X55))`
- 预测能力(IC): -0.0640
- 稳定性: 0.5408
- 多头收益: 0.0006
- 空头收益: 0.0011
- 复杂度: 7

##### 因子 2:

- 表达式: `div(add(X3, X14), mul(X37, X18))`
- 预测能力(IC): -0.0107
- 稳定性: 0.4865
- 多头收益: -0.0002
- 空头收益: 0.0015
- 复杂度: 7

##### 因子 3:

- 表达式: `div(sin(X57), mul(X49, X29))`
- 预测能力(IC): -0.0317
- 稳定性: 0.3258
- 多头收益: -0.0009
- 空头收益: -0.0006
- 复杂度: 6

##### 因子 4:

- 表达式: `abs(div(log(X27), abs(X50)))`
- 预测能力(IC): -0.0094
- 稳定性: 0.9844
- 多头收益: -0.0010
- 空头收益: -0.0029
- 复杂度: 6

##### 因子 5:

- 表达式: `div(abs(mul(X1, X31)), mul(mul(X25, X58), mul(X32, X21)))`
- 预测能力(IC): -0.0032
- 稳定性: 0.1189
- 多头收益: 0.0001
- 空头收益: -0.0008
- 复杂度: 12

