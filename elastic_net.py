import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, save_model

# ENet模型调参
def tune_elastic_net(X_train, y_train):
    # 简化参数网格，减少搜索范围
    estimator = ElasticNet()
    param_grid = {'l1_ratio': [0.1, 0.5, 0.9, 1.0], 'alpha': [0.1, 0.5, 1.0]}
    grid = GridSearchCV(estimator, param_grid, scoring='neg_mean_squared_error', cv=2, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_params_

# 训练ElasticNet模型
def train_elastic_net(X_train, y_train, X_test, y_test):
    # 使用最佳参数
    best_params = tune_elastic_net(X_train, y_train)
    print(f"ElasticNet最佳参数: {best_params}")
    
    elnet = ElasticNet(**best_params)
    print("开始训练ElasticNet模型...")
    elnet.fit(X_train, y_train)
    print("ElasticNet模型训练完成！")
    y_pred_elnet = elnet.predict(X_test)
    
    # 计算评估指标
    elnet_res = evaluate_model(y_pred_elnet, y_test, "ElasticNet")
    
    # 保存模型
    save_model(elnet, "ElasticNet")
    
    return elnet_res