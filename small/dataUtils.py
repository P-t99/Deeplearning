import pandas as pd
import numpy as np
import re
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d



def get_raw_data_from_csv(filepath, label, normalization=False, data_augmentation=False):

    dataframes = pd.read_csv(filepath)
    x = []
    y = []

    for i in dataframes.index:
        data = dataframes.loc[i]
        item_1 = list(map(float, re.findall(r"\d+\.?\d*", data['data'])))
        item = np.asarray(item_1)
        x.append(item.astype(np.float32))
        if data[label] == 1:
            y.append(0)
        elif data[label] == 2:
            y.append(1)
        elif data[label] == 3:
            y.append(2)
    x = np.array(x).astype(np.float32)
    y = np.array(y)
    return x, y


def get_data(filepath, label, normalization=False, shuffle=True, train_ratio=0.8, data_augmentation='False'):

    x, y = get_raw_data_from_csv(filepath, label, normalization, data_augmentation)

    if shuffle:
        permutation = np.random.permutation(y.shape[0])
        x = x[permutation]
        y = y[permutation]

    train_total = x.shape[0]
    train_nums = int(train_total * train_ratio)

    x_train = x[0:train_nums]
    y_train = y[0:train_nums]

    x_test = x[train_nums:train_total]
    y_test = y[train_nums:train_total]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    data_path = "sdata.csv"
    get_data(data_path, 'position', train_ratio=0.8, data_augmentation='True')
