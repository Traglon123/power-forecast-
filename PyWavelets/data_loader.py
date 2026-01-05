import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def get_australia_data(data_path, window_size=24, target_col='Power load'):
    # --- 第一步：读取文件 ---
    # encoding='utf-8' 是为了防止读取英文文件时出错
    df = pd.read_csv(data_path, encoding='utf-8')

    # --- 第二步：挑选出“数字”列 ---
    # 原始数据里有时间字符串，模型不认识，我们只要负荷、气温这些数字
    df_numeric = df.select_dtypes(include=[np.number])

    # 确保我们要预测的那一列（比如 TOTALDEMAND）在最前面
    # 这样 y 始终取第一列就不会错
    cols = [target_col] + [c for c in df_numeric.columns if c != target_col]
    data = df_numeric[cols].values

    # 3. 归一化处理
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 4. 构建时间窗口数据集
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        # 比如 window_size 是 24，就取 0~23 行的数据作为输入
        X.append(scaled_data[i : i + window_size, :])
        # 取第 24 行的第一列（负荷）作为预测目标
        y.append(scaled_data[i + window_size, 0])

    # --- 第五步：转换成 PyTorch 能用的格式 ---
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).view(-1, 1) # 变成一列

    return X, y, scaler