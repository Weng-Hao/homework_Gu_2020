import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, save_model

# 神经网络模型调参
def tune_nn(X_train, y_train, hidden_layer_sizes):
    # 简化调参过程，直接返回固定参数
    return {'learning_rate_init': 0.001}

# 训练神经网络模型
def train_nn(X_train, y_train, X_test, y_test, hidden_layer_sizes, model_name):
    # 使用最佳参数
    best_params = tune_nn(X_train, y_train, hidden_layer_sizes)
    print(f"{model_name}最佳参数: {best_params}")
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
            
    # 根据模型不同设置不同的alpha值
    if model_name in ['NN1', 'NN2']:
        alpha = 0.001
    else:
        alpha = 0.1
    
    # 根据模型不同设置不同的max_iter值
    if model_name in ['NN1', 'NN5']:
        max_iter = 5000
    else:
        max_iter = 2000
    
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, 
        activation='relu', 
        solver='adam', 
        alpha=alpha, 
        max_iter=1000,  # 减少迭代次数
        learning_rate_init=best_params['learning_rate_init'], 
        random_state=12
    )
    print(f"开始训练{model_name}模型...")
    mlp.fit(X_train, y_train)
    print(f"{model_name}模型训练完成！")
    y_pred_mlp = mlp.predict(X_test)
    
    # 计算评估指标
    mlp_res = evaluate_model(y_pred_mlp, y_test, model_name)
    
    # 保存模型
    save_model(mlp, model_name)
    
    return mlp_res

# 训练所有神经网络模型
def train_all_nn(X_train, y_train, X_test, y_test):
    # 定义不同的神经网络结构
    nn_configs = {
        #'NN1': (16,),
        #'NN2': (16, 8),
        #'NN3': (16, 8, 4),
        #'NN4': (16, 8, 4, 2),
        'NN5': (16, 8, 4, 2, 1)
    }
    
    results = {}
    for model_name, hidden_layer_sizes in nn_configs.items():
        print(f"\n训练{model_name}模型:")
        results[model_name] = train_nn(X_train, y_train, X_test, y_test, hidden_layer_sizes, model_name)
    
    return results