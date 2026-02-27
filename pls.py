import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, save_model

# 训练PLS模型
def train_pls(X_train, y_train, X_test, y_test):
    # PLS模型调参
    pls_model_setup = PLSRegression(scale=True)
    param_grid = {'n_components': [1, 5, 10]}  # 简化参数搜索范围
    search = GridSearchCV(pls_model_setup, param_grid, verbose=1, n_jobs=-1)
    
    # 训练模型
    print("开始训练PLS模型...")
    pls_model = search.fit(X_train, y_train)
    print("PLS模型训练完成！")
    print(f"PLS最佳参数: {{'n_components': {pls_model.best_params_['n_components']}}}")
    
    # 预测
    pls_prediction = pls_model.predict(X_test)
    y_pred_pls = pls_prediction.reshape(-1,)
    
    # 计算评估指标
    pls_res = evaluate_model(y_pred_pls, y_test, "PLS")
    
    # 保存模型
    save_model(pls_model, "PLS")
    
    return pls_res