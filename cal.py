import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入AlphasSingleStock类
from alpha101_code_1 import *

# 读取数据
def load_data():
    return pd.read_parquet('daily_data.parquet')

# 利用alpha101_code_1里面的get_alpha函数计算单个股票的Alpha因子

# 主函数
def main():
    # 加载数据
    print("Loading data...")
    data = load_data()
    data = data.sort_index()
    # 按股票分组计算Alpha因子
    print("Calculating Alpha factors...")
    alpha_results = []
    
    # 获取所有股票代码
    symbols = data.index.get_level_values('order_book_id').unique()
    total_symbols = len(symbols)
    
    for i, symbol in enumerate(symbols):
        if i % 10 == 0:
            print(f"Processing stock {i+1}/{total_symbols}: {symbol}")
        
        # 获取单只股票的数据
        df_stock = data.loc[data.index.get_level_values('order_book_id') == symbol].copy()
        
        # 重置索引，只保留日期作为索引
        df_stock = df_stock.reset_index(level='order_book_id')
        
        # 计算Alpha因子
        alpha_df = get_alpha(df_stock)
        
        # 重新设置multiindex
        alpha_df['order_book_id'] = symbol
        alpha_df = alpha_df.set_index('order_book_id', append=True)
        alpha_df = alpha_df.reorder_levels(['date', 'order_book_id'])
        
        # 添加到结果列表
        alpha_results.append(alpha_df)
    
    # 组合所有结果
    print("Combining results...")
    final_result = pd.concat(alpha_results)
    
    # 保存结果
    print("Saving results...")
    final_result.to_parquet('alpha101_results_1000.parquet')
    print("Done!")
    
    return final_result

if __name__ == '__main__':
    result = main()
    print("Result shape:", result.shape)
    print("First few rows:")
    print(result.head())