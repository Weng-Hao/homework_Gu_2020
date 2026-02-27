import numpy as np
import statsmodels.api as sm
from utils import evaluate_model, save_model

# 训练OLS模型
def train_ols(X_train, y_train, X_test, y_test):
    # 添加常数项
    X_train_o = sm.add_constant(X_train)
    X_test_o = sm.add_constant(X_test)
    
    # 拟合模型
    print("开始训练OLS模型...")
    ols = sm.OLS(y_train, X_train_o).fit()
    print("OLS模型训练完成！")
    print("OLS模型摘要:")
    print(ols.summary())
    print()
    
    # 预测
    y_pred_ols = ols.predict(X_test_o)
    
    # 计算评估指标
    ols_res = evaluate_model(y_pred_ols, y_test, "OLS")
    
    # 保存模型
    save_model(ols, "OLS")
    
    return ols_res