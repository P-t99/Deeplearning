# 导入所需的库
import os  # 用于文件和目录操作，如路径拼接和文件遍历
import numpy as np  # 用于数值计算，特别是矩阵和数组操作
import pandas as pd  # 用于数据处理和分析，提供了DataFrame数据结构
import matplotlib.pyplot as plt  # 用于绘图，特别是数据的可视化
import logging  # 用于记录日志信息，方便调试和监控代码运行情况
from sklearn.linear_model import LogisticRegression  # 用于逻辑回归模型的训练
from sklearn.model_selection import train_test_split  # 用于将数据集划分为训练集和测试集
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # 用于模型评估，包括混淆矩阵、分类报告和准确率
from matplotlib import font_manager  # 用于字体管理，特别是设置中文字体

# 设置中文字体
# font_path = r'C:\Windows\Fonts\simhei.ttf'  # 指定中文字体文件路径
# font_prop = font_manager.FontProperties(fname=font_path)  # 创建字体属性对象，用于管理字体
# plt.rcParams['font.family'] = font_prop.get_name()  # 设置matplotlib全局字体为指定的中文字体


# 设置全局字体为 'SimHei'（黑体），这在很多系统中都有包含
plt.rcParams['font.family'] = ['SimHei'] #run commands,运行时参数
# 设置负号正常显示
plt.rcParams['axes.unicode_minus'] = False


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # 配置日志记录的基本设置，指定日志级别和格式

# 向量化操作池化函数
def vectorized_pooling(matrix, pool_size):
    """
    对矩阵进行池化操作
    :param matrix: 原始矩阵
    :param pool_size: 池化窗口大小，格式为 (mxn)
    :return: 池化后的矩阵
    """
    m, n = pool_size
    h, w = matrix.shape
    
    # 计算新的高度和宽度
    new_h = (h // m) * m
    new_w = (w // n) * n
    
    # 计算需要裁剪的行和列
    crop_h = (h - new_h) // 2
    crop_w = (w - new_w) // 2
    
    # 调整矩阵大小，使其能被池化窗口整除
    matrix = matrix[crop_h:crop_h + new_h, crop_w:crop_w + new_w]
    
    # 计算池化后的矩阵
    pooled_matrix = matrix.reshape(new_h // m, m, new_w // n, n).mean(axis=(1, 3))
    
    return pooled_matrix

def safe_parse_matrix(data_str):
    """
    安全地解析字符串形式的矩阵数据
    :param data_str: 字符串形式的矩阵数据
    :return: 解析后的numpy数组，如果解析失败则返回None
    """
    try:
        # 将字符串转换为浮点数列表，移除字符串中的空格并按逗号分隔
        values = [float(x.strip()) for x in data_str.strip('[]').replace(' ', '').split(',') if x.strip()]
        if len(values) == 1024:
            matrix = np.array(values).reshape(32, 32)
            return vectorized_pooling(matrix, (2, 3))  # 池化为16x10
        elif len(values) == 160:
            return np.array(values).reshape(16, 10)
        else:
            raise ValueError(f"预期1024或160个元素，但得到{len(values)}个")
    except Exception as e:
        logging.error(f"解析矩阵时出错: {e}")  # 记录解析错误
        return None  # 返回None表示解析失败

def normalize_matrix(matrix):
    """
    对矩阵进行归一化处理
    :param matrix: 输入矩阵
    :return: 归一化后的矩阵
    """
    min_val = np.min(matrix)  # 获取矩阵中的最小值
    max_val = np.max(matrix)  # 获取矩阵中的最大值
    if np.isclose(max_val, min_val):
        logging.warning("矩阵的最大值和最小值相等，跳过归一化")  # 如果最大值和最小值相等，记录警告并跳过归一化
        return matrix  # 直接返回原矩阵
    return (matrix - min_val) / (max_val - min_val)  # 按公式对矩阵进行归一化处理

def load_and_preprocess_data(folder_path, label):
    """
    加载并预处理指定文件夹中的csv数据
    :param folder_path: 数据文件夹路径
    :param label: 数据标签
    :return: 处理后的数据列表和对应的标签列表
    """
    data = []  # 用于存储处理后的数据
    labels = []  # 用于存储对应的标签
    for filename in os.listdir(folder_path):  # 遍历文件夹中的所有文件
        if filename.endswith('.csv'):  # 仅处理.csv文件
            file_path = os.path.join(folder_path, filename)  # 构造文件的完整路径
            try:
                df = pd.read_csv(file_path)  # 读取CSV文件为DataFrame
                for _, row in df.iterrows():  # 逐行遍历DataFrame
                    matrix = safe_parse_matrix(row['data'])  # 解析数据列中的矩阵字符串
                    if matrix is not None:
                        # normalized_matrix = normalize_matrix(matrix)  # 对矩阵进行归一化处理
                        normalized_matrix = matrix
                        data.append(normalized_matrix)  # 将归一化后的矩阵添加到数据列表中
                        labels.append(label)  # 将对应的标签添加到标签列表中
            except Exception as e:
                logging.error(f"处理文件 {filename} 时出错: {e}")  # 记录文件处理错误
    logging.info(f"从文件夹 {folder_path} 加载了 {len(data)} 个样本，标签为 {label}")  # 记录加载数据的数量和标签
    return data, labels  # 返回处理后的数据列表和标签列表



def extract_features(matrix):
    """
    从矩阵中提取特征
    :param matrix: 输入矩阵
    :return: 提取的特征列表
    """
    flat_matrix = matrix.flatten()  # 将矩阵展平为一维数组
    
    # 计算矩阵元素的80分位数

    percentile_80 = np.mean(flat_matrix) + 0.8*np.std(flat_matrix)
    
    
    # 特征1: topN大且大于80分位数的点所在的唯一行的个数
    N = 32
    topN_indices = np.argsort(flat_matrix)[-N:]  # 获取数组中最大N个值的索引
    topN_values = flat_matrix[topN_indices]
    valid_topN_indices = topN_indices[topN_values > percentile_80]  # 只保留大于80分位数的点
    valid_topN_rows = valid_topN_indices // matrix.shape[1]  # 通过整数除法获取这些值所在的行
    unique_valid_topN_rows = np.unique(valid_topN_rows)  # 获取唯一的行索引
    
    return [len(unique_valid_topN_rows)]

def train_model(X, y):
    """
    训练逻辑回归模型并评估性能
    :param X: 特征矩阵
    :param y: 标签向量
    :return: 训练好的模型
    """
    # y_train:训练集的真实标签，y_test:测试集的真实标签
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分数据集为训练集和测试集，比例为8:2

    model = LogisticRegression(random_state=42)  # 创建逻辑回归模型对象
    model.fit(X_train, y_train)  # 使用训练集训练模型

    y_pred = model.predict(X_test)  # 使用模型对测试集进行预测

    train_accuracy = accuracy_score(y_train, model.predict(X_train))  # 计算训练集的准确率
    test_accuracy = accuracy_score(y_test, y_pred)  # 计算测试集的准确率

    # 输出模型性能指标
    logging.info("=" * 50)
    logging.info("训练集指标:")
    logging.info(f"训练集准确度: {train_accuracy:.4f}")
    logging.info(f"测试集准确度: {test_accuracy:.4f}")
    logging.info("测试集 Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))  # 打印测试集的混淆矩阵
    logging.info("测试集 Classification Report:\n%s", classification_report(y_test, y_pred))  # 打印测试集的分类报告

    # 计算并输出特征重要性
    feature_importance = np.abs(model.coef_[0])  # 获取模型的系数（特征的重要性）
    feature_names = ['列数特征']  # 定义特征的名称
    for name, importance in zip(feature_names, feature_importance):  # 遍历特征名称和重要性
        logging.info(f"特征 '{name}' 的重要性: {importance:.4f}")

    logging.info("=" * 50)

    return model  # 返回训练好的模型

def get_model_parameters(model):
    """
    获取模型的参数（系数和截距）
    :param model: 训练好的模型
    :return: 系数和截距
    """
    coefficients = model.coef_[0]  # 获取模型的系数
    intercept = model.intercept_[0]  # 获取模型的截距
    return coefficients, intercept  # 返回系数和截距

def embedded_system_logic(features, coefficients, intercept):
    """
    模拟嵌入式系统的逻辑，计算预测概率
    :param features: 输入特征
    :param coefficients: 模型系数
    :param intercept: 模型截距
    :return: 预测概率
    """
    log_odds = np.dot(features, coefficients) + intercept  # 计算log-odds，即线性模型的输出
    probability = 1 / (1 + np.exp(-log_odds))  # 应用sigmoid函数将log-odds转换为概率
    return probability  # 返回预测概率


def test_embedded_model(test_sit_folder, test_fall_folder, coefficients, intercept):
    """
    测试嵌入式模型的性能
    :param test_sit_folder: 坐起测试数据文件夹
    :param test_fall_folder: 坠床测试数据文件夹
    :param coefficients: 模型系数
    :param intercept: 模型截距
    :return: 预测概率列表和真实标签列表
    """
    logging.info("开始嵌入式模型测试...")
    test_sit_data, test_sit_labels = load_and_preprocess_data(test_sit_folder, label="坐起")  # 加载和预处理坐起测试数据
    test_fall_data, test_fall_labels = load_and_preprocess_data(test_fall_folder, label="坠床风险")  # 加载和预处理坠床测试数据

    test_data = test_sit_data + test_fall_data  # 合并坐起和坠床的数据
    test_labels = test_sit_labels + test_fall_labels  # 合并坐起和坠床的标签

    correct = 0  # 初始化正确预测计数
    total = len(test_data)  # 总样本数

    y_true = []  # 真实标签列表
    y_pred = []  # 预测标签列表
    probabilities = []  # 预测概率列表

    for matrix, label in zip(test_data, test_labels):  # 遍历每个测试样本及其标签
        features = extract_features(matrix)  # 提取特征
        probability = embedded_system_logic(features, coefficients, intercept)  # 计算预测概率
        probabilities.append(probability)  # 将预测概率添加到列表中
        prediction = "坠床风险" if probability > 0.5 else "坐起"  # 基于概率进行预测
        if prediction == label:
            correct += 1  # 如果预测正确，增加计数

        y_true.append(0 if label == "坐起" else 1)  # 将标签转换为数值（0: 坐起, 1: 坠床风险）
        y_pred.append(0 if prediction == "坐起" else 1)  # 将预测结果转换为数值

    # 计算并输出性能指标
    accuracy = correct / total  # 计算预测准确率
    logging.info("=" * 50)
    logging.info("嵌入式模型测试指标:")
    logging.info(f"嵌入式模型测试准确率: {accuracy:.4f}")
    logging.info("嵌入式模型测试 Confusion Matrix:\n%s", confusion_matrix(y_true, y_pred))  # 打印混淆矩阵
    logging.info("嵌入式模型测试 Classification Report:\n%s", classification_report(y_true, y_pred))  # 打印分类报告
    logging.info("=" * 50)

    return probabilities, y_true  # 返回预测概率列表和真实标签列表


def plot_prediction_probabilities(probabilities, y_true):
    """
    绘制预测概率分布图
    :param probabilities: 预测概率列表
    :param y_true: 真实标签列表
    """
    plt.figure(figsize=(10, 6))  # 设置图形尺寸
    plt.scatter(range(len(probabilities)), probabilities, c=y_true, cmap='coolwarm')  # 绘制散点图，颜色表示真实标签
    plt.colorbar(label='真实标签 (0: 坐起, 1: 坠床风险)')  # 添加颜色条，标注标签含义
    plt.xlabel('样本索引')  # 设置x轴标签
    plt.ylabel('预测概率')  # 设置y轴标签
    plt.title('嵌入式模型预测概率分布')  # 设置图形标题
    plt.axhline(y=0.5, color='r', linestyle='--')  # 添加一条横线，表示决策边界
    plt.text(len(probabilities), 0.5, '决策边界', va='center', ha='left', backgroundcolor='w')  # 在决策边界处添加文本
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig('prediction_probabilities.png')  # 将图像保存为文件
    logging.info("预测概率图像已保存为 'prediction_probabilities.png'")  # 记录图像保存信息
    plt.close()  # 关闭图形，释放内存

def plot_75th_percentile_curve(test_data):
    """
    绘制测试数据集中每个矩阵中元素的1.5倍均值逐帧曲线图
    :param test_data: 测试数据集
    """
    means = [np.mean(matrix)+0.9*np.std(matrix) for matrix in test_data]  # 计算每个矩阵的1.5倍均值
    plt.figure(figsize=(10, 6))  # 设置图形尺寸
    plt.plot(means, marker='o')  # 绘制逐帧曲线图
    plt.xlabel('样本索引')  # 设置x轴标签
    plt.ylabel('1.5倍均值')  # 设置y轴标签
    plt.title('测试数据集1.5倍均值逐帧曲线图')  # 设置图形标题
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig('1.5_mean_curve.png')  # 将图像保存为文件
    logging.info("1.5倍均值逐帧曲线图已保存为 '1.5_mean_curve.png'")  # 记录图像保存信息
    plt.close()  # 关闭图形，释放内存

def main():
    """
    主函数，执行整个工作流程
    """
    # 设置数据路径
    base_path = r'D:\repository\Algo_test\dataL'
    
    def get_data_paths(data_type='train'):
        """
        获取指定类型的数据路径
        :param data_type: 'train' 或 'test'
        :return: 坐起和坠床数据的文件夹路径
        """
        folder = 'train_data' if data_type == 'train' else 'test_data'
        sit_folder = os.path.join(base_path, folder, '坐起')
        fall_folder = os.path.join(base_path, folder, '坠床')
        return sit_folder, fall_folder
    
    # 获取训练数据路径
    sit_bed_folder, fall_bed_folder = get_data_paths('test')
    
    # 获取测试数据路径
    test_sit_bed_folder, test_fall_bed_folder = get_data_paths('train')

    # 加载和预处理训练数据
    logging.info("开始加载和预处理训练数据...")
    sit_data, sit_labels = load_and_preprocess_data(sit_bed_folder, label="坐起")  # 加载坐起数据
    fall_data, fall_labels = load_and_preprocess_data(fall_bed_folder, label="坠床风险")  # 加载坠床数据

    all_data = sit_data + fall_data  # 合并所有数据
    all_labels = sit_labels + fall_labels  # 合并所有标签

    logging.info(f"总共加载了 {len(all_data)} 个训练样本")  # 记录加载的总样本数
    logging.info(f"训练数据类别分布: 坐起 - {all_labels.count('坐起')}, 坠床风险 - {all_labels.count('坠床风险')}")  # 记录类别分布

    # 特征提取
    logging.info("开始特征提取...")
    X = [extract_features(matrix) for matrix in all_data]  # 提取所有数据的特征
    y = [0 if label == "坐起" else 1 for label in all_labels]  # 将标签转换为数值

    # 模型训练
    logging.info("开始模型训练...")
    model = train_model(X, y)  # 训练模型

    # 获取模型参数
    logging.info("获取模型参数...")
    coefficients, intercept = get_model_parameters(model)  # 获取模型的系数和截距
    logging.info("嵌入式系统使用的系数: %s", coefficients)  # 记录系数
    logging.info("嵌入式系统使用的截距: %s", intercept)  # 记录截距

    # 嵌入式模型测试
    logging.info("开始嵌入式模型测试...")
    probabilities, y_true = test_embedded_model(test_sit_bed_folder, test_fall_bed_folder, coefficients, intercept)  # 测试嵌入式模型

    # 绘制预测概率图像
    logging.info("绘制预测概率图像...")
    plot_prediction_probabilities(probabilities, y_true)  # 绘制并保存预测概率分布图

    # 绘制1.5倍均值逐帧曲线图
    logging.info("绘制1.5倍均值逐帧曲线图...")
    plot_75th_percentile_curve(all_data)  # 绘制并保存1.5倍均值逐帧曲线图
    

if __name__ == "__main__":
    main()  # 调用主函数，执行整个流程
