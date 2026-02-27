import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
def load_data():
    # 加载alpha101结果和 daily_data
    alpha_df = pd.read_parquet('alpha101_results_1000.parquet')
    
    alpha_df['next_day_return'] = alpha_df['close']/alpha_df['prev_close'] - 1
    
    
    # 处理无穷值和NaN值
    alpha_df = alpha_df.replace([np.inf, -np.inf], np.nan)
    alpha_df = alpha_df.dropna()
    
    return alpha_df

# 预处理数据
def preprocess_data(df):
    # 划分训练集和测试集
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=123)
    
    # 准备模型输入和输出数据
    # 选择alpha因子作为特征（假设alpha因子列名以alpha开头）
    alpha_columns = [col for col in df.columns if col.startswith('alpha')]
    X_train = df_train[alpha_columns]   # 训练集输入特征（alpha因子）
    y_train = df_train['next_day_return'].values.flatten()  # 训练集输出标签（next_day_return）
    X_test = df_test[alpha_columns]  # 测试集输入特征  
    y_test = df_test['next_day_return'].values.flatten()  # 测试集输出标签
    
    # 数据标准化处理
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)  # 对训练集进行拟合和标准化
    X_test_scaled = scale.transform(X_test)  # 使用训练集的参数对测试集进行标准化
    
    return X_train_scaled, y_train, X_test_scaled, y_test, df_train, df_test