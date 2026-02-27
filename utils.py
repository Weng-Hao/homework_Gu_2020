import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# 定义ROOS（Ratio of Out-of-Sample Explained Variance）评估指标
def roos(y_pre, y_true):
    up = np.sum((y_true - y_pre) ** 2)  # 计算预测值与真实值之差的平方和
    down = np.sum(y_true ** 2)  # 计算真实值的平方和
    return (1 - up / down)/100  # 返回ROOS值，越接近1越好

# 计算并打印评估指标
def evaluate_model(y_pred, y_test, model_name):
    mse = mean_squared_error(y_pred, y_test)  # 均方误差
    mae = mean_absolute_error(y_pred, y_test)  # 平均绝对误差
    rmse = mse ** 0.5  # 均方根误差
    roos2 = roos(y_pred, y_test)  # ROOS值
    r2 = r2_score(y_test, y_pred)  # R²评分
    
    print(f"{model_name}评估指标:")
    print(f'mse: {mse}, rmse: {rmse}, mae: {mae}, roos2: {roos2}')
    print(f'r2_score: {r2}')
    print()
    
    return [mse, rmse, mae, roos2]

# 保存模型
def save_model(model, model_name):
    try:
        joblib.dump(model, f'{model_name}.joblib')
        print(f"{model_name}模型已保存到 {model_name}.joblib")
    except Exception as e:
        print(f"保存{model_name}模型时出错: {e}")