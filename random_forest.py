import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from utils import evaluate_model, save_model

# 随机森林模型调参

def tune_random_forest(X_train, y_train):
    # 简化参数网格，减少搜索范围
    estimator = RandomForestRegressor(random_state=1, n_jobs=-1)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 15, 20]}
    # 减少交叉验证折数，提高速度
    grid = GridSearchCV(estimator, param_grid, scoring='neg_mean_squared_error', cv=2, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_params_

# 固定超参数的简单调参函数
def tune_simple():
    # 返回固定的超参数
    return {'n_estimators': 150, 'max_depth': 15}

def train_random_forest(X_train, y_train, X_test, y_test, df_train):
    # 使用固定超参数
    best_params = tune_simple()
    print(f"随机森林使用固定参数: {best_params}")
    
    rf = RandomForestRegressor(**best_params, random_state=0, n_jobs=-1)
    print("开始训练随机森林模型...")
    
    # 训练模型（移除进度条相关代码，直接训练）
    rf.fit(X_train, y_train)
    print("随机森林模型训练完成！")
    
    y_pred_rf = rf.predict(X_test)
    
    # 计算评估指标
    rf_res = evaluate_model(y_pred_rf, y_test, "随机森林")
    
    print("随机森林特征重要性:")
    print(rf.feature_importances_)
    print()
    
    # 保存模型
    save_model(rf, "RandomForest")
    
    return rf_res