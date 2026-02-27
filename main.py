import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
from data_preprocessing import load_data, preprocess_data
from random_forest import train_random_forest
from gbrt import train_gbrt
from neural_network import train_all_nn
from pls import train_pls
from elastic_net import train_elastic_net
from ols import train_ols
from pcr import train_pcr
from lstm import train_lstm
from utils import evaluate_model

# 主函数
def main():
    # 加载数据
    df = load_data()
    
    # 预处理数据
    X_train, y_train, X_test, y_test, df_train, df_test = preprocess_data(df)
    
    print("数据加载和预处理完成")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print()

    # 训练OLS模型
    # 读取已训练好的OLS模型
    if os.path.exists('OLS.joblib'):
        print("\n=== 读取已训练好的OLS模型 ===")
        ols_model = joblib.load('OLS.joblib')
        print("OLS模型读取成功！")
        # 评估模型
        import statsmodels.api as sm
        X_test_o = sm.add_constant(X_test)
        y_pred_ols = ols_model.predict(X_test_o)
        ols_res = evaluate_model(y_pred_ols, y_test, "OLS")
    else:
        print("\n=== 训练OLS模型 ===")
        ols_res = train_ols(X_train, y_train, X_test, y_test)
    
    # 训练PCR模型
    # 读取已训练好的PCR模型
    if os.path.exists('PCR.joblib'):
        print("\n=== 读取已训练好的PCR模型 ===")
        pcr_model = joblib.load('PCR.joblib')
        print("PCR模型读取成功！")
        # 评估模型
        y_pred_pcr = pcr_model.predict(X_test)
        pols_res = evaluate_model(y_pred_pcr, y_test, "PCR")
    else:
        print("\n=== 训练PCR模型 ===")
        pols_res = train_pcr(X_train, y_train, X_test, y_test)

    # 训练PLS模型
    # 读取已训练好的PLS模型
    if os.path.exists('PLS.joblib'):
        print("\n=== 读取已训练好的PLS模型 ===")
        pls_model = joblib.load('PLS.joblib')
        print("PLS模型读取成功！")
        # 评估模型
        pls_prediction = pls_model.predict(X_test)
        y_pred_pls = pls_prediction.reshape(-1,)
        pls_res = evaluate_model(y_pred_pls, y_test, "PLS")
    else:
        print("\n=== 训练PLS模型 ===")
        pls_res = train_pls(X_train, y_train, X_test, y_test)
    
    # 训练ElasticNet模型
    # 读取已训练好的ElasticNet模型
    if os.path.exists('ElasticNet.joblib'):
        print("\n=== 读取已训练好的ElasticNet模型 ===")
        elnet_model = joblib.load('ElasticNet.joblib')
        print("ElasticNet模型读取成功！")
        # 评估模型
        y_pred_elnet = elnet_model.predict(X_test)
        elnet_res = evaluate_model(y_pred_elnet, y_test, "ElasticNet")
    else:
        print("\n=== 训练ElasticNet模型 ===")
        elnet_res = train_elastic_net(X_train, y_train, X_test, y_test)

    # 训练LSTM模型
    # 读取已训练好的LSTM模型
    if os.path.exists('LSTM.h5'):
        print("\n=== 读取已训练好的LSTM模型 ===")
        from tensorflow.keras.models import load_model
        lstm_model = load_model('LSTM.h5')
        print("LSTM模型读取成功！")
        # 评估模型
        from lstm import prepare_lstm_data
        import numpy as np
        time_steps = 1
        _, _, X_test_lstm, y_test_lstm, scaler_y = prepare_lstm_data(X_train, y_train, X_test, y_test, time_steps)
        y_pred_scaled = lstm_model.predict(X_test_lstm)
        y_pred_lstm = scaler_y.inverse_transform(y_pred_scaled).reshape(-1,)
        y_true = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1)).reshape(-1,)
        lstm_res = evaluate_model(y_pred_lstm, y_true, "LSTM")
    else:
        print("\n=== 训练LSTM模型 ===")
        lstm_res = train_lstm(X_train, y_train, X_test, y_test, time_steps=1)

    # 训练随机森林模型
    # 读取已训练好的随机森林模型
    if os.path.exists('RandomForest.joblib'):
        print("\n=== 读取已训练好的随机森林模型 ===")
        try:
            rf_model = joblib.load('RandomForest.joblib')
            print("随机森林模型读取成功！")
            # 评估模型
            y_pred_rf = rf_model.predict(X_test)
            rf_res = evaluate_model(y_pred_rf, y_test, "Random Forest")
        except Exception as e:
            print(f"随机森林模型读取失败: {e}")
            print("\n=== 训练随机森林模型 ===")
            rf_res = train_random_forest(X_train, y_train, X_test, y_test, df_train)
    else:
        print("\n=== 训练随机森林模型 ===")
        rf_res = train_random_forest(X_train, y_train, X_test, y_test, df_train)
    

    
    # 训练GBRT模型
    if os.path.exists('GBRT.joblib'):
        print("\n=== 读取已训练好的GBRT模型 ===")
        try:
            gbrt_model = joblib.load('GBRT.joblib')
            print("GBRT模型读取成功！")
            # 评估模型
            y_pred_gbrt = gbrt_model.predict(X_test)
            gbrt_res = evaluate_model(y_pred_gbrt, y_test, "GBRT")
        except Exception as e:
            print(f"GBRT模型读取失败: {e}")
            print("\n=== 训练GBRT模型 ===")
            gbrt_res = train_gbrt(X_train, y_train, X_test, y_test, df_train)
    else:
        print("\n=== 训练GBRT模型 ===")
        gbrt_res = train_gbrt(X_train, y_train, X_test, y_test, df_train) 
    

    # 训练神经网络模型
    nn_models = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']
    nn_res_dict = {}
    # 先检查是否存在已训练的神经网络模型
    all_nn_exist = True
    for model_name in nn_models:
        if not os.path.exists(f'{model_name}.joblib'):
            all_nn_exist = False
            break
    
    if all_nn_exist:
        print("\n=== 读取已训练好的神经网络模型 ===")
        try:
            for model_name in nn_models:
                print(f"\n=== 读取已训练好的{model_name}模型 ===")
                nn_model = joblib.load(f'{model_name}.joblib')
                print(f"{model_name}模型读取成功！")
                # 评估模型
                y_pred_nn = nn_model.predict(X_test)
                nn_res = evaluate_model(y_pred_nn, y_test, model_name)
                nn_res_dict[model_name] = nn_res
            mlp1_res = nn_res_dict['NN1']
            mlp2_res = nn_res_dict['NN2']
            mlp3_res = nn_res_dict['NN3']
            mlp4_res = nn_res_dict['NN4']
            mlp5_res = nn_res_dict['NN5']
        except Exception as e:
            print(f"神经网络模型读取失败: {e}")
            print("\n=== 训练神经网络模型 ===")
            nn_results = train_all_nn(X_train, y_train, X_test, y_test)
            mlp1_res = nn_results['NN1']
            mlp2_res = nn_results['NN2']
            mlp3_res = nn_results['NN3']
            mlp4_res = nn_results['NN4']
            mlp5_res = nn_results['NN5']
    else:
        print("\n=== 训练神经网络模型 ===")
        nn_results = train_all_nn(X_train, y_train, X_test, y_test)
        mlp1_res = nn_results['NN1']
        mlp2_res = nn_results['NN2']
        mlp3_res = nn_results['NN3']
        mlp4_res = nn_results['NN4']
        mlp5_res = nn_results['NN5']
    

    
    # 汇总所有模型的评估结果
    print("\n=== 模型评估结果汇总 ===")
    res = pd.DataFrame([
        rf_res, gbrt_res, mlp1_res, mlp2_res, mlp3_res, mlp4_res, mlp5_res, 
        pls_res, elnet_res, ols_res, pols_res, lstm_res
    ])
    res.columns = ['mse', 'rmse', 'mae', 'roos2']
    res.index = ['Random Forest', 
                'GBRT', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'PLS', 'ENet', 'OLS', 'PCR', 'LSTM']
    
    print(res)
    print()
    
    # 保存结果到Excel文件
    res.to_excel('模型评估结果.xlsx')

if __name__ == "__main__":
    main()