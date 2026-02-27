import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, save_model

# 训练PCR模型
def train_pcr(X_train, y_train, X_test, y_test):
    # 创建管道
    regression_model = LinearRegression(normalize=True)
    pca_model = PCA()
    pipe = Pipeline(steps=[('pca', pca_model), ('least_squares', regression_model)])
    
    # 调参
    param_grid = {'pca__n_components': [1, 5, 10]}  # 简化参数搜索范围
    search = GridSearchCV(pipe, param_grid, verbose=1, n_jobs=-1)
    
    # 训练模型
    print("开始训练PCR模型...")
    pcareg_model = search.fit(X_train, y_train)
    print("PCR模型训练完成！")
    print(f"PCR最佳参数: {{'n_components': {pcareg_model.best_params_['pca__n_components']}}}")
    
    # 预测
    pcareg_prediction = pcareg_model.predict(X_test)
    
    # 计算评估指标
    pols_res = evaluate_model(pcareg_prediction, y_test, "PCR")
    
    # 保存模型
    save_model(pcareg_model, "PCR")
    
    return pols_res