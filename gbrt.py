import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from utils import evaluate_model, save_model

# GBRT模型调参
# 固定超参数的简单调参函数
def tune_simple():
    # 返回固定的超参数
    return {'n_estimators': 150, 'learning_rate': 0.1}

# 训练GBRT模型
def train_gbrt(X_train, y_train, X_test, y_test, df_train):
    # 使用固定超参数
    best_params = tune_simple()
    print(f"GBRT使用固定参数: {best_params}")
    
    # 处理缺失值：使用中位数填充
    print("检查并处理NaN...")
    if hasattr(X_train, 'isnull'):
        # pandas DataFrame
        if X_train.isnull().any().any():
            print(f"训练集发现缺失值，数量: {X_train.isnull().sum().sum()}，已用0填充")
            X_train = X_train.fillna(0)
        if X_test.isnull().any().any():
            print(f"测试集发现缺失值，数量: {X_test.isnull().sum().sum()}，已用0填充")
            X_test = X_test.fillna(0)
    else:
        # numpy array
        if np.isnan(X_train).any():
            print(f"训练集发现缺失值，数量: {np.isnan(X_train).sum()}，已用0填充")
            X_train = np.nan_to_num(X_train, nan=0)
        if np.isnan(X_test).any():
            print(f"测试集发现缺失值，数量: {np.isnan(X_test).sum()}，已用0填充")
            X_test = np.nan_to_num(X_test, nan=0)
    
    gbrt = GradientBoostingRegressor(**best_params, random_state=1)
    print("开始训练GBRT模型...")
    
    # 训练模型（移除进度条循环，直接训练）
    gbrt.fit(X_train, y_train)
    print("GBRT模型训练完成！")
    
    y_pred_gbrt = gbrt.predict(X_test)
    
    # 计算评估指标
    gbrt_res = evaluate_model(y_pred_gbrt, y_test, "GBRT")
    

    print("GBRT特征重要性:")
    print(gbrt.feature_importances_)
    print()
    
    # 保存模型
    save_model(gbrt, "GBRT")
    
    return gbrt_res