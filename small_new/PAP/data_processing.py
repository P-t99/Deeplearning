# data_processing.py

import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于科学计算和数组操作
import re  # 用于正则表达式操作

# 导入train_test_split函数用于数据集的划分
from sklearn.model_selection import train_test_split

def get_raw_data_from_csv(filepath, label, normalization=False, data_augmentation=False):
    # 从CSV文件读取数据
    dataframes = pd.read_csv(filepath)  # 如果没有列名，则header=None
    x = []
    y = []

    # 遍历数据框的每一行
    for i in dataframes.index:
        data = dataframes.loc[i]
        # 从'data'列提取数值并转换为浮点数
        item_1 = list(map(float, re.findall(r"\d+\.?\d*", data['data'])))
        item = np.asarray(item_1)
        # 将特征数据添加到x列表
        x.append(item.astype(np.float32))
        # 根据标签值设置对应的类别
        if data[label] == 1:
            y.append(0)
        elif data[label] == 2:
            y.append(1)
        elif data[label] == 3:
            y.append(2)
        elif data[label] == 4:
            y.append(3)
    
    # 将列表转换为numpy数组
    x = np.array(x).astype(np.float32) #np.asarray 可能更有效率，如果本来是array，则不复制
    y = np.array(y)
    return x, y

def get_data(filepath, label, normalization=False, shuffle=True, test_size=0.2, data_augmentation='False'):
    # 获取原始数据
    x, y = get_raw_data_from_csv(filepath, label, normalization, data_augmentation)

    if shuffle:
        # 使用sklearn的train_test_split函数进行分割
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True, stratify=y)
    else:
        # 如果不需要打乱，直接按比例分割
        train_total = x.shape[0]
        test_nums = int(train_total * test_size)
        x_train, x_test = x[:-test_nums], x[-test_nums:]
        y_train, y_test = y[:-test_nums], y[-test_nums:]

    return x_train, y_train, x_test, y_test

# 写一个调用此模块的示例
if __name__ == "__main__":
    filepath = r'data.csv'
    label = 'position'
    x_train, y_train, x_test, y_test = get_data(filepath, label)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)