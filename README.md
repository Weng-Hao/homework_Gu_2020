# 回归模型训练与评估项目

本项目实现了多种回归模型的训练与评估，包括随机森林、GBRT、多层神经网络、PLS、ElasticNet、OLS和PCR等模型。

## 项目结构

```
├── data_preprocessing.py  # 数据预处理模块
├── random_forest.py    # 随机森林模型
├── gbrt.py             # 梯度提升回归树模型
├── neural_network.py   # 神经网络模型（NN1-NN5）
├── lstm.py             # LSTM模型
├── pls.py              # 偏最小二乘模型
├── elastic_net.py      # 弹性网络模型
├── ols.py              # 普通最小二乘模型
├── pcr.py              # 主成分回归模型
├── alpha101_code_1.py  # Alpha 101因子计算模块
├── cal.py              # Alpha因子计算主文件
├── utils.py            # 工具函数（评估指标计算等）
├── main.py             # 主运行文件
├── daily_data.parquet  # 原始数据文件
├── alpha101_results_1000.parquet  # Alpha因子计算结果
├── OLS.joblib          # 训练好的OLS模型
├── PCR.joblib          # 训练好的PCR模型
├── PLS.joblib          # 训练好的PLS模型
├── ElasticNet.joblib   # 训练好的ElasticNet模型
├── RandomForest.joblib # 训练好的随机森林模型
├── GBRT.joblib         # 训练好的GBRT模型
├── NN1.joblib          # 训练好的NN1模型
├── NN2.joblib          # 训练好的NN2模型
├── NN3.joblib          # 训练好的NN3模型
├── NN4.joblib          # 训练好的NN4模型
├── NN5.joblib          # 训练好的NN5模型
└── LSTM.h5             # 训练好的LSTM模型
```

## 功能说明

### 1. 数据预处理模块 (`data_preprocessing.py`)
- 加载数据集
- 划分训练集和测试集（80%训练，20%测试）
- 特征标准化处理

### 2. 模型模块

#### 随机森林模型 (`random_forest.py`)
- 实现随机森林回归
- 网格搜索调参（n_estimators, max_features）
- 特征重要性分析和可视化

#### GBRT模型 (`gbrt.py`)
- 实现梯度提升回归树
- 网格搜索调参（n_estimators, learning_rate, max_depth）
- 特征重要性分析和可视化

#### 神经网络模型 (`neural_network.py`)
- 实现5种不同结构的神经网络（NN1-NN5）
- 网格搜索调参（alpha, max_iter）
- 不同网络结构的参数设置

#### PLS模型 (`pls.py`)
- 实现偏最小二乘回归
- 网格搜索调参（n_components）

#### ElasticNet模型 (`elastic_net.py`)
- 实现弹性网络回归
- 网格搜索调参（l1_ratio, alpha）

#### OLS模型 (`ols.py`)
- 实现普通最小二乘回归
- 输出模型摘要

#### PCR模型 (`pcr.py`)
- 实现主成分回归
- 网格搜索调参（n_components）

#### LSTM模型 (`lstm.py`)
- 实现长短期记忆网络回归
- 数据归一化处理
- 构建两层LSTM网络结构
- 训练参数：epochs=50, batch_size=32

### 3. 工具模块 (`utils.py`)
- 实现ROOS评估指标计算
- 实现模型评估指标计算和打印（MSE, RMSE, MAE, ROOS²）

### 4. 主运行文件 (`main.py`)
- 加载和预处理数据
- 依次运行所有模型
- 汇总评估结果并保存到Excel文件
- 绘制不同模型的评估指标对比图

### 5. Alpha 101因子计算模块 (`alpha101_code_1.py`)
- 实现AlphasSingleStock类，用于计算101个Alpha因子
- 包含各种技术指标计算函数（如ts_sum、sma、stddev等）
- 实现101个Alpha因子的具体计算逻辑

### 6. Alpha因子计算主文件 (`cal.py`)
- 加载daily_data.parquet数据文件
- 遍历每只股票计算Alpha因子
- 合并结果为MultiIndex格式
- 保存计算结果到alpha101_results_1000.parquet文件

## 依赖库

- pandas
- numpy
- scikit-learn (0.24.2)
- statsmodels
- matplotlib
- tensorflow
- scipy
- joblib

## 使用方法

### 1. 运行回归模型
1. 确保所有依赖库已安装
2. 运行主文件：
   ```bash
   python main.py
   ```
3. 运行结果将显示在控制台
4. 注意事项：
   - 首次运行时会训练所有模型，可能需要较长时间
   - 后续运行会尝试加载已训练好的模型
   - 若模型文件版本不兼容，会自动重新训练
   - 随机森林和GBRT模型训练时间较长
   - 可能会出现SciPy与NumPy版本兼容性警告，不影响运行

### 2. 计算Alpha因子
1. 确保所有依赖库已安装
2. 运行Alpha因子计算文件：
   ```bash
   python cal.py
   ```
3. 运行结果将显示在控制台，并保存到"alpha101_results_1000.parquet"文件中

## 评估指标

- MSE (Mean Squared Error)：均方误差
- RMSE (Root Mean Squared Error)：均方根误差
- MAE (Mean Absolute Error)：平均绝对误差
- ROOS² (R-squared Out-of-Sample)：样本外R方

## 模型结构

### 神经网络结构
- NN1: 单隐藏层（32个神经元）
- NN2: 单隐藏层（64个神经元）
- NN3: 双隐藏层（64, 32个神经元）
- NN4: 双隐藏层（128, 64个神经元）
- NN5: 三隐藏层（128, 64, 32个神经元）

### LSTM模型结构
- 输入层：时间步长为20
- 第一层LSTM：50个神经元，返回序列
- Dropout：20%
- 第二层LSTM：50个神经元，不返回序列
- Dropout：20%
- 输出层：1个神经元

## 结果分析

运行完成后，将在控制台显示所有模型的评估指标表格，可用于比较不同模型的性能。


### 模型选择建议
- 基于ROOS²指标，OLS、PLS和ElasticNet模型表现较好
- 随机森林和GBRT模型表现较好，但训练时间较长
- LSTM和NN5出现了严重的过拟合，在样本外表现较弱

根据具体应用场景和计算资源，选择最适合的模型。