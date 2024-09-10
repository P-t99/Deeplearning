import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

def improved_current_distribution_correction(matrix, alpha=0.5, correct_rows=True, correct_cols=True):
    # 计算行和列的总和
    row_sums = matrix.sum(axis=1, keepdims=True)
    col_sums = matrix.sum(axis=0, keepdims=True)
    
    # 初始化校正因子
    row_factors = np.ones_like(matrix)
    col_factors = np.ones_like(matrix)
    
    # 如果需要对行进行校正
    if correct_rows:
        row_factors = 1 / (1 + alpha * (matrix / row_sums))
    
    # 如果需要对列进行校正
    if correct_cols:
        col_factors = 1 / (1 + alpha * (matrix / col_sums))
    
    # 应用校正
    corrected_matrix = matrix * row_factors * col_factors
    
    # 重新调整以保持行和列的总和不变
    if correct_rows:
        row_correction = row_sums / corrected_matrix.sum(axis=1, keepdims=True)
    else:
        row_correction = np.ones_like(row_sums)
    
    if correct_cols:
        col_correction = col_sums / corrected_matrix.sum(axis=0, keepdims=True)
    else:
        col_correction = np.ones_like(col_sums)
    
    final_matrix = corrected_matrix * np.sqrt(row_correction * col_correction)
    
    return final_matrix

def plot_heatmaps(matrix_data, alpha=0.5, correct_rows=True, correct_cols=True):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 将输入数据转换为NumPy数组并重塑为16x10的矩阵
    data = np.array(matrix_data)
    original_matrix = data.reshape(16, 10)
    
    # 应用电流分布校正
    corrected_matrix = improved_current_distribution_correction(original_matrix, alpha, correct_rows, correct_cols)

    # 创建原始坐标网格
    x = np.arange(0, 10)
    y = np.arange(0, 16)

    # 创建更精细的网格进行插值
    x_new = np.linspace(0, 9, 50)
    y_new = np.linspace(0, 15, 80)
    xx, yy = np.meshgrid(x_new, y_new)

    # 定义不同的插值方法
    methods = {
        '原始数据': None,
        '线性插值': RegularGridInterpolator,
        'B样条插值': RectBivariateSpline
    }

    # 创建两个图形，每个图形包含3个子图，调整总窗口大小
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    correction_type = []
    if correct_rows and correct_cols:
        correction_type = "行列校正"
    elif correct_rows:
        correction_type = "仅行校正"
    elif correct_cols:
        correction_type = "仅列校正"
    else:
        correction_type = "无校正"

    for fig, axes, matrix, title in [(fig1, axes1, original_matrix, '未校正'), 
                                     (fig2, axes2, corrected_matrix, f'校正后 ({correction_type})')]:
        for ax, (method_name, method) in zip(axes, methods.items()):
            if method is None:
                im = ax.imshow(matrix, origin='upper', aspect=1.2, cmap='viridis')
            elif method == RegularGridInterpolator:
                interp = method((y, x), matrix)
                points = np.array([yy.flatten(), xx.flatten()]).T
                z_new = interp(points).reshape(80, 50)
                im = ax.imshow(z_new, origin='upper', aspect=1.2, cmap='viridis', extent=[0, 9, 15, 0])
            elif method == RectBivariateSpline:
                interp = method(y, x, matrix)
                z_new = interp(y_new, x_new)
                im = ax.imshow(z_new, origin='upper', aspect=1.2, cmap='viridis', extent=[0, 9, 15, 0])

            ax.set_title(f'{method_name}')
            ax.set_xlabel('X轴')
            ax.set_ylabel('Y轴')

        fig.colorbar(im, ax=axes.ravel().tolist(), label='压力值')
        fig.suptitle(f'压力分布热图 - {title}', fontsize=16)

    plt.show()

# 使用示例
matrix_data = [0,1,5,9,21,24,18,19,35,3,0,1,1,4,18,19,41,22,21,2,2,17,29,42,55,50,36,19,10,4,3,4,2,39,61,58,31,3,1,8,42,1,1,11,20,25,27,2,1,7,35,1,1,26,28,29,30,5,2,9,2,2,2,18,39,40,38,32,2,2,9,3,2,22,34,24,28,32,11,22,9,0,3,31,18,17,30,19,3,6,0,0,3,28,7,2,21,15,1,1,0,0,1,28,2,1,13,4,0,0,1,0,2,7,0,0,3,5,0,0,0,1,40,23,2,0,3,36,2,0,0,1,40,14,1,0,6,45,2,0,0,0,24,2,0,1,1,22,1,0,0,0,2,0,0,0,1,7,0,2]

# 绘制校正前和校正后的热图，可以控制对行、列是否进行电流补偿校正
plot_heatmaps(matrix_data, alpha=10, correct_rows=False, correct_cols=True)
