# 因子网格搜索报告

生成时间: 2025-03-28 02:23:37

## 总体统计

- 参数组合总数: 13
- 成功完成组合: 13
- 失败组合: 0

## 最佳因子 (按IC排序)

### 因子 1

- 表达式: `div(X2, -0.136)`
- 预测能力(IC): 0.1459
- 稳定性: 0.9992
- 多头收益: 0.0041
- 空头收益: 0.0017
- 复杂度: 3
- 参数组合:
  - forward_period: 36
  - population_size: 1000
  - generations: 100
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 2

- 表达式: `div(neg(log(X3)), neg(abs(X35)))`
- 预测能力(IC): 0.0325
- 稳定性: 0.6709
- 多头收益: 0.0006
- 空头收益: 0.0005
- 复杂度: 7
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 100
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 3

- 表达式: `neg(div(div(X39, X19), sqrt(X42)))`
- 预测能力(IC): 0.0253
- 稳定性: 0.5622
- 多头收益: 0.0008
- 空头收益: 0.0005
- 复杂度: 7
- 参数组合:
  - forward_period: 30
  - population_size: 3000
  - generations: 100
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 4

- 表达式: `div(div(log(-0.872), mul(X17, X48)), sqrt(mul(X41, X44)))`
- 预测能力(IC): 0.0239
- 稳定性: 0.3789
- 多头收益: 0.0006
- 空头收益: 0.0005
- 复杂度: 11
- 参数组合:
  - forward_period: 18
  - population_size: 1000
  - generations: 300
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 5

- 表达式: `abs(sub(sub(mul(sub(div(X41, X5), div(X41, X17)), log(div(X56, X48))), sub(div(mul(X11, X24), log(X21)), add(sub(X43, X33), div(X13, X9)))), add(mul(neg(mul(X57, X56)), sub(cos(X57), abs(X12))), mul(log(sqrt(X42)), sqrt(cos(X33))))))`
- 预测能力(IC): 0.0233
- 稳定性: 0.1025
- 多头收益: 0.0013
- 空头收益: -0.0006
- 复杂度: 47
- 参数组合:
  - forward_period: 30
  - population_size: 3000
  - generations: 100
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 6

- 表达式: `neg(div(X57, X50))`
- 预测能力(IC): 0.0231
- 稳定性: 0.6919
- 多头收益: 0.0003
- 空头收益: -0.0004
- 复杂度: 4
- 参数组合:
  - forward_period: 18
  - population_size: 1000
  - generations: 300
  - tournament_size: 30
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 7

- 表达式: `sub(X7, div(X19, X14))`
- 预测能力(IC): 0.0220
- 稳定性: 0.7013
- 多头收益: 0.0011
- 空头收益: -0.0011
- 复杂度: 5
- 参数组合:
  - forward_period: 30
  - population_size: 3000
  - generations: 100
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 8

- 表达式: `div(sin(X47), mul(X11, X27))`
- 预测能力(IC): 0.0220
- 稳定性: 0.7843
- 多头收益: 0.0003
- 空头收益: 0.0014
- 复杂度: 6
- 参数组合:
  - forward_period: 30
  - population_size: 3000
  - generations: 100
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 9

- 表达式: `div(mul(sin(X57), abs(X22)), mul(mul(X36, X61), mul(X48, X7)))`
- 预测能力(IC): 0.0203
- 稳定性: 0.0533
- 多头收益: 0.0003
- 空头收益: 0.0001
- 复杂度: 13
- 参数组合:
  - forward_period: 12
  - population_size: 1000
  - generations: 100
  - tournament_size: 10
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

### 因子 10

- 表达式: `mul(div(sqrt(sub(X20, X31)), mul(sin(X56), cos(X21))), sub(div(div(X8, X46), neg(X30)), cos(mul(X8, X22))))`
- 预测能力(IC): 0.0175
- 稳定性: 0.0928
- 多头收益: -0.0005
- 空头收益: -0.0002
- 复杂度: 22
- 参数组合:
  - forward_period: 30
  - population_size: 3000
  - generations: 100
  - tournament_size: 50
  - p_crossover: 0.5
  - p_subtree_mutation: 0.2
  - p_hoist_mutation: 0.1
  - p_point_mutation: 0.1
  - parsimony_coefficient: 0.001
  - ic_threshold: 0.05
  - stability_threshold: 0.3
  - min_long_return: 0.5
  - min_short_return: -0.5
  - enable_segment_test: True
  - test_set_ratio: 0.3
  - max_complexity: 20

## 各参数组合结果

### 组合 1

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 100
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 84.25秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X17, X48)`
- 预测能力(IC): -0.0429
- 稳定性: 0.5940
- 多头收益: -0.0001
- 空头收益: -0.0002
- 复杂度: 3

##### 因子 2:

- 表达式: `mul(cos(div(mul(sub(X56, X35), neg(X15)), abs(sub(X56, X19)))), add(mul(sqrt(neg(X60)), div(cos(X41), abs(X59))), div(cos(sqrt(X20)), log(log(X38)))))`
- 预测能力(IC): -0.0219
- 稳定性: 0.6972
- 多头收益: -0.0005
- 空头收益: -0.0010
- 复杂度: 30

##### 因子 3:

- 表达式: `div(mul(sin(X57), abs(X22)), mul(mul(X36, X61), mul(X48, X7)))`
- 预测能力(IC): 0.0203
- 稳定性: 0.0533
- 多头收益: 0.0003
- 空头收益: 0.0001
- 复杂度: 13

##### 因子 4:

- 表达式: `div(sin(log(mul(X51, X24))), log(abs(div(X31, X28))))`
- 预测能力(IC): -0.0034
- 稳定性: 0.0643
- 多头收益: 0.0002
- 空头收益: 0.0001
- 复杂度: 11

##### 因子 5:

- 表达式: `mul(div(neg(X10), neg(X32)), neg(div(X54, X60)))`
- 预测能力(IC): -0.0008
- 稳定性: -0.0612
- 多头收益: 0.0010
- 空头收益: -0.0004
- 复杂度: 10

### 组合 2

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 100
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 85.00秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X42, X60)`
- 预测能力(IC): -0.0152
- 稳定性: 0.1869
- 多头收益: 0.0003
- 空头收益: -0.0004
- 复杂度: 3

##### 因子 2:

- 表达式: `add(abs(X44), div(X5, X48))`
- 预测能力(IC): -0.0112
- 稳定性: 0.2972
- 多头收益: -0.0000
- 空头收益: -0.0001
- 复杂度: 6

##### 因子 3:

- 表达式: `div(X8, X59)`
- 预测能力(IC): -0.0100
- 稳定性: 0.0396
- 多头收益: 0.0001
- 空头收益: -0.0000
- 复杂度: 3

##### 因子 4:

- 表达式: `neg(add(neg(mul(sub(log(X52), cos(X14)), mul(abs(X36), mul(X59, X61)))), mul(add(log(sub(X33, X13)), add(sqrt(X38), sub(X51, X50))), div(log(sub(X61, X55)), mul(sin(X21), cos(X31))))))`
- 预测能力(IC): 0.0013
- 稳定性: 0.1795
- 多头收益: 0.0003
- 空头收益: -0.0011
- 复杂度: 37

##### 因子 5:

- 表达式: `div(div(log(X43), log(X41)), cos(log(X1)))`
- 预测能力(IC): 0.0020
- 稳定性: 0.0107
- 多头收益: -0.0001
- 空头收益: -0.0000
- 复杂度: 9

### 组合 3

#### 参数:

- forward_period: 12
- population_size: 1000
- generations: 100
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 85.54秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(neg(log(X3)), neg(abs(X35)))`
- 预测能力(IC): 0.0325
- 稳定性: 0.6709
- 多头收益: 0.0006
- 空头收益: 0.0005
- 复杂度: 7

##### 因子 2:

- 表达式: `div(X11, 0.017)`
- 预测能力(IC): -0.0761
- 稳定性: 0.9918
- 多头收益: -0.0008
- 空头收益: -0.0016
- 复杂度: 3

##### 因子 3:

- 表达式: `sub(X18, div(neg(sin(X50)), mul(neg(-0.134), sqrt(X46))))`
- 预测能力(IC): -0.0657
- 稳定性: 0.9581
- 多头收益: -0.0007
- 空头收益: -0.0015
- 复杂度: 11

##### 因子 4:

- 表达式: `div(log(X4), neg(X3))`
- 预测能力(IC): -0.0170
- 稳定性: 0.0043
- 多头收益: -0.0004
- 空头收益: -0.0002
- 复杂度: 5

##### 因子 5:

- 表达式: `neg(div(X0, X58))`
- 预测能力(IC): -0.0006
- 稳定性: 0.0080
- 多头收益: 0.0004
- 空头收益: -0.0001
- 复杂度: 4

### 组合 4

#### 参数:

- forward_period: 12
- population_size: 3000
- generations: 100
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 237.34秒

#### 找到的因子:

##### 因子 1:

- 表达式: `add(mul(sub(sin(cos(X25)), add(sub(X25, X14), sub(X39, X16))), sub(div(sqrt(X24), mul(X42, X18)), add(add(X27, X2), neg(X15)))), mul(neg(add(add(X29, X3), mul(X39, X1))), sin(add(div(X45, X60), div(X32, X1)))))`
- 预测能力(IC): -0.0128
- 稳定性: 0.9012
- 多头收益: -0.0005
- 空头收益: -0.0008
- 复杂度: 43

##### 因子 2:

- 表达式: `div(sub(log(add(X16, X20)), div(log(X14), sub(X47, X27))), sin(cos(cos(X3))))`
- 预测能力(IC): 0.0008
- 稳定性: 0.5640
- 多头收益: 0.0007
- 空头收益: 0.0001
- 复杂度: 16

##### 因子 3:

- 表达式: `mul(div(mul(X7, X11), neg(X12)), div(cos(X57), log(X40)))`
- 预测能力(IC): 0.0021
- 稳定性: 0.1203
- 多头收益: 0.0006
- 空头收益: -0.0003
- 复杂度: 12

##### 因子 4:

- 表达式: `div(mul(sqrt(add(X2, X27)), mul(add(X18, X13), div(X0, X33))), mul(sin(cos(X50)), sqrt(log(X54))))`
- 预测能力(IC): -0.0085
- 稳定性: 0.0319
- 多头收益: 0.0003
- 空头收益: 0.0000
- 复杂度: 20

##### 因子 5:

- 表达式: `div(div(X59, X17), X58)`
- 预测能力(IC): 0.0010
- 稳定性: 0.0003
- 多头收益: 0.0004
- 空头收益: -0.0006
- 复杂度: 5

### 组合 5

#### 参数:

- forward_period: 12
- population_size: 3000
- generations: 100
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 242.83秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(div(cos(X31), sin(X56)), sin(div(X13, X55)))`
- 预测能力(IC): -0.0180
- 稳定性: -0.0299
- 多头收益: 0.0000
- 空头收益: -0.0005
- 复杂度: 10

### 组合 6

#### 参数:

- forward_period: 18
- population_size: 1000
- generations: 300
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 247.87秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(div(log(-0.872), mul(X17, X48)), sqrt(mul(X41, X44)))`
- 预测能力(IC): 0.0239
- 稳定性: 0.3789
- 多头收益: 0.0006
- 空头收益: 0.0005
- 复杂度: 11

##### 因子 2:

- 表达式: `div(X52, X51)`
- 预测能力(IC): -0.0327
- 稳定性: 0.9037
- 多头收益: 0.0002
- 空头收益: -0.0013
- 复杂度: 3

##### 因子 3:

- 表达式: `div(X44, X51)`
- 预测能力(IC): -0.0158
- 稳定性: 0.8957
- 多头收益: -0.0006
- 空头收益: -0.0008
- 复杂度: 3

##### 因子 4:

- 表达式: `add(abs(sin(log(X44))), add(div(log(X9), log(X11)), cos(neg(X16))))`
- 预测能力(IC): -0.0032
- 稳定性: 0.5388
- 多头收益: 0.0009
- 空头收益: -0.0008
- 复杂度: 14

##### 因子 5:

- 表达式: `sub(div(abs(cos(X19)), abs(sub(X31, X49))), sqrt(cos(sub(X51, X59))))`
- 预测能力(IC): 0.0054
- 稳定性: 0.4222
- 多头收益: 0.0006
- 空头收益: -0.0005
- 复杂度: 14

### 组合 7

#### 参数:

- forward_period: 18
- population_size: 1000
- generations: 300
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 267.74秒

#### 找到的因子:

##### 因子 1:

- 表达式: `neg(div(X57, X50))`
- 预测能力(IC): 0.0231
- 稳定性: 0.6919
- 多头收益: 0.0003
- 空头收益: -0.0004
- 复杂度: 4

##### 因子 2:

- 表达式: `neg(div(add(abs(sin(X60)), div(mul(X55, X51), div(X14, X29))), mul(sub(cos(X28), abs(X48)), add(abs(X37), add(X21, X14)))))`
- 预测能力(IC): -0.0133
- 稳定性: 0.1393
- 多头收益: 0.0000
- 空头收益: -0.0010
- 复杂度: 25

##### 因子 3:

- 表达式: `div(X54, X6)`
- 预测能力(IC): -0.0152
- 稳定性: 0.0014
- 多头收益: 0.0012
- 空头收益: -0.0005
- 复杂度: 3

##### 因子 4:

- 表达式: `div(X6, X58)`
- 预测能力(IC): 0.0108
- 稳定性: -0.0056
- 多头收益: 0.0001
- 空头收益: 0.0001
- 复杂度: 3

##### 因子 5:

- 表达式: `div(cos(X45), X17)`
- 预测能力(IC): 0.0015
- 稳定性: 0.0338
- 多头收益: -0.0000
- 空头收益: -0.0006
- 复杂度: 4

### 组合 8

#### 参数:

- forward_period: 30
- population_size: 3000
- generations: 100
- tournament_size: 10
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 231.44秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(sin(X47), mul(X11, X27))`
- 预测能力(IC): 0.0220
- 稳定性: 0.7843
- 多头收益: 0.0003
- 空头收益: 0.0014
- 复杂度: 6

##### 因子 2:

- 表达式: `sub(X7, div(X19, X14))`
- 预测能力(IC): 0.0220
- 稳定性: 0.7013
- 多头收益: 0.0011
- 空头收益: -0.0011
- 复杂度: 5

##### 因子 3:

- 表达式: `div(sub(X12, X19), neg(X59))`
- 预测能力(IC): -0.0257
- 稳定性: 0.0577
- 多头收益: 0.0001
- 空头收益: 0.0001
- 复杂度: 6

##### 因子 4:

- 表达式: `abs(sub(sub(mul(sub(div(X41, X5), div(X41, X17)), log(div(X56, X48))), sub(div(mul(X11, X24), log(X21)), add(sub(X43, X33), div(X13, X9)))), add(mul(neg(mul(X57, X56)), sub(cos(X57), abs(X12))), mul(log(sqrt(X42)), sqrt(cos(X33))))))`
- 预测能力(IC): 0.0233
- 稳定性: 0.1025
- 多头收益: 0.0013
- 空头收益: -0.0006
- 复杂度: 47

##### 因子 5:

- 表达式: `div(div(X30, X14), div(X9, X36))`
- 预测能力(IC): -0.0004
- 稳定性: 0.1326
- 多头收益: 0.0008
- 空头收益: -0.0003
- 复杂度: 7

### 组合 9

#### 参数:

- forward_period: 30
- population_size: 3000
- generations: 100
- tournament_size: 30
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 238.41秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(abs(sub(sqrt(X11), add(X54, X15))), div(neg(mul(X25, X48)), div(sin(X44), sqrt(X8))))`
- 预测能力(IC): -0.0186
- 稳定性: 0.4191
- 多头收益: 0.0007
- 空头收益: 0.0007
- 复杂度: 18

##### 因子 2:

- 表达式: `div(neg(div(div(X5, X34), mul(X47, X40))), div(sub(mul(X45, X54), mul(X60, X14)), sin(cos(X48))))`
- 预测能力(IC): 0.0097
- 稳定性: -0.2144
- 多头收益: 0.0003
- 空头收益: 0.0003
- 复杂度: 20

##### 因子 3:

- 表达式: `div(div(div(X56, X47), mul(X11, X27)), div(div(X6, X6), mul(X14, X21)))`
- 预测能力(IC): 0.0100
- 稳定性: 0.4538
- 多头收益: -0.0000
- 空头收益: -0.0001
- 复杂度: 15

##### 因子 4:

- 表达式: `div(add(0.467, X10), mul(X29, X22))`
- 预测能力(IC): 0.0026
- 稳定性: 0.4622
- 多头收益: -0.0002
- 空头收益: -0.0015
- 复杂度: 7

##### 因子 5:

- 表达式: `div(cos(sub(cos(abs(X44)), cos(div(X4, X37)))), mul(mul(abs(cos(X30)), sqrt(sub(X11, X41))), neg(sub(abs(X37), abs(X9)))))`
- 预测能力(IC): 0.0024
- 稳定性: 0.0573
- 多头收益: 0.0005
- 空头收益: -0.0001
- 复杂度: 25

### 组合 10

#### 参数:

- forward_period: 30
- population_size: 3000
- generations: 100
- tournament_size: 50
- p_crossover: 0.5
- p_subtree_mutation: 0.2
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 237.27秒

#### 找到的因子:

##### 因子 1:

- 表达式: `neg(div(div(X39, X19), sqrt(X42)))`
- 预测能力(IC): 0.0253
- 稳定性: 0.5622
- 多头收益: 0.0008
- 空头收益: 0.0005
- 复杂度: 7

##### 因子 2:

- 表达式: `div(abs(abs(X52)), div(mul(X30, X24), cos(X21)))`
- 预测能力(IC): -0.0068
- 稳定性: 0.6983
- 多头收益: 0.0013
- 空头收益: 0.0007
- 复杂度: 10

##### 因子 3:

- 表达式: `add(neg(X42), div(div(X34, X6), X20))`
- 预测能力(IC): -0.0183
- 稳定性: 0.0385
- 多头收益: 0.0010
- 空头收益: -0.0003
- 复杂度: 8

##### 因子 4:

- 表达式: `mul(div(sqrt(sub(X20, X31)), mul(sin(X56), cos(X21))), sub(div(div(X8, X46), neg(X30)), cos(mul(X8, X22))))`
- 预测能力(IC): 0.0175
- 稳定性: 0.0928
- 多头收益: -0.0005
- 空头收益: -0.0002
- 复杂度: 22

##### 因子 5:

- 表达式: `sub(mul(sub(sub(X31, X16), sub(X4, X49)), div(div(X61, X59), neg(X28))), X7)`
- 预测能力(IC): -0.0013
- 稳定性: -0.0220
- 多头收益: -0.0003
- 空头收益: -0.0007
- 复杂度: 16

### 组合 11

#### 参数:

- forward_period: 36
- population_size: 1000
- generations: 100
- tournament_size: 50
- p_crossover: 0.7
- p_subtree_mutation: 0.15
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 77.87秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X2, -0.136)`
- 预测能力(IC): 0.1459
- 稳定性: 0.9992
- 多头收益: 0.0041
- 空头收益: 0.0017
- 复杂度: 3

##### 因子 2:

- 表达式: `div(X25, X43)`
- 预测能力(IC): -0.0175
- 稳定性: 0.7327
- 多头收益: -0.0000
- 空头收益: -0.0027
- 复杂度: 3

##### 因子 3:

- 表达式: `neg(abs(sub(div(X21, X58), log(X44))))`
- 预测能力(IC): 0.0108
- 稳定性: 0.1105
- 多头收益: 0.0007
- 空头收益: -0.0000
- 复杂度: 8

##### 因子 4:

- 表达式: `sub(div(X1, X4), X33)`
- 预测能力(IC): 0.0100
- 稳定性: 0.0171
- 多头收益: 0.0018
- 空头收益: -0.0019
- 复杂度: 5

##### 因子 5:

- 表达式: `div(sub(mul(X48, X47), mul(X31, X47)), div(mul(X54, X59), mul(X46, X39)))`
- 预测能力(IC): 0.0084
- 稳定性: 0.0089
- 多头收益: 0.0005
- 空头收益: 0.0001
- 复杂度: 15

### 组合 12

#### 参数:

- forward_period: 36
- population_size: 1000
- generations: 300
- tournament_size: 10
- p_crossover: 0.7
- p_subtree_mutation: 0.15
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 244.22秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X28, X32)`
- 预测能力(IC): 0.0090
- 稳定性: 0.3776
- 多头收益: -0.0009
- 空头收益: -0.0010
- 复杂度: 3

##### 因子 2:

- 表达式: `div(X4, X51)`
- 预测能力(IC): -0.0054
- 稳定性: 0.5818
- 多头收益: -0.0002
- 空头收益: -0.0006
- 复杂度: 3

##### 因子 3:

- 表达式: `div(div(log(div(X52, X10)), add(sin(X3), sub(X9, X16))), sin(log(sqrt(X15))))`
- 预测能力(IC): -0.0112
- 稳定性: 0.1491
- 多头收益: 0.0006
- 空头收益: -0.0016
- 复杂度: 16

##### 因子 4:

- 表达式: `div(X47, X4)`
- 预测能力(IC): -0.0241
- 稳定性: 0.0071
- 多头收益: 0.0011
- 空头收益: -0.0014
- 复杂度: 3

##### 因子 5:

- 表达式: `div(X31, X6)`
- 预测能力(IC): 0.0136
- 稳定性: 0.0027
- 多头收益: 0.0009
- 空头收益: -0.0011
- 复杂度: 3

### 组合 13

#### 参数:

- forward_period: 36
- population_size: 1000
- generations: 300
- tournament_size: 30
- p_crossover: 0.7
- p_subtree_mutation: 0.15
- p_hoist_mutation: 0.1
- p_point_mutation: 0.1
- parsimony_coefficient: 0.001
- ic_threshold: 0.05
- stability_threshold: 0.3
- min_long_return: 0.5
- min_short_return: -0.5
- enable_segment_test: True
- test_set_ratio: 0.3
- max_complexity: 20

#### 运行时间: 248.05秒

#### 找到的因子:

##### 因子 1:

- 表达式: `div(X2, X30)`
- 预测能力(IC): -0.0389
- 稳定性: 0.9197
- 多头收益: 0.0014
- 空头收益: 0.0001
- 复杂度: 3

##### 因子 2:

- 表达式: `div(X6, X50)`
- 预测能力(IC): -0.0013
- 稳定性: -0.1040
- 多头收益: -0.0001
- 空头收益: -0.0002
- 复杂度: 3

##### 因子 3:

- 表达式: `div(X17, X56)`
- 预测能力(IC): -0.0098
- 稳定性: 0.0131
- 多头收益: 0.0003
- 空头收益: -0.0010
- 复杂度: 3

##### 因子 4:

- 表达式: `div(X23, X57)`
- 预测能力(IC): -0.0049
- 稳定性: 0.0009
- 多头收益: -0.0004
- 空头收益: -0.0012
- 复杂度: 3

##### 因子 5:

- 表达式: `add(div(X39, X57), abs(X56))`
- 预测能力(IC): -0.0002
- 稳定性: 0.0099
- 多头收益: 0.0007
- 空头收益: -0.0012
- 复杂度: 6

