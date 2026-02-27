import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from utils import evaluate_model
import numpy as np

# 准备LSTM数据
def prepare_lstm_data(X_train, y_train, X_test, y_test, time_steps=1):
    # 对数据进行归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # 转换为LSTM所需的时间序列格式 [samples, time_steps, features]
    X_train_lstm = []
    y_train_lstm = []
    for i in range(time_steps, len(X_train_scaled)):
        X_train_lstm.append(X_train_scaled[i-time_steps:i, :])
        y_train_lstm.append(y_train_scaled[i, 0])
    
    X_test_lstm = []
    y_test_lstm = []
    for i in range(time_steps, len(X_test_scaled)):
        X_test_lstm.append(X_test_scaled[i-time_steps:i, :])
        y_test_lstm.append(y_test_scaled[i, 0])
    
    # 转换为numpy数组
    X_train_lstm = np.array(X_train_lstm)
    y_train_lstm = np.array(y_train_lstm)
    X_test_lstm = np.array(X_test_lstm)
    y_test_lstm = np.array(y_test_lstm)
    
    return X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler_y

# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 训练LSTM模型
def train_lstm(X_train, y_train, X_test, y_test, time_steps=1):
    # 准备LSTM数据
    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler_y = prepare_lstm_data(
        X_train, y_train, X_test, y_test, time_steps
    )
    
    print(f"LSTM数据准备完成，训练集形状: {X_train_lstm.shape}, 测试集形状: {X_test_lstm.shape}")
    
    # 构建模型
    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
    model = build_lstm_model(input_shape)
    
    # 训练模型
    print("开始训练LSTM模型...")
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=20, batch_size=64,  # 减少训练轮数，增加批量大小
        validation_data=(X_test_lstm, y_test_lstm),
        verbose=1
    )
    print("LSTM模型训练完成！")
    
    # 预测
    y_pred_scaled = model.predict(X_test_lstm)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1,)
    y_true = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1)).reshape(-1,)
    
    # 计算评估指标
    lstm_res = evaluate_model(y_pred, y_true, "LSTM")
    
    # 保存模型
    try:
        model.save('LSTM.h5')
        print("LSTM模型已保存到 LSTM.h5")
    except Exception as e:
        print(f"保存LSTM模型时出错: {e}")
    
    return lstm_res