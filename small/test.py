# 导入必要的库
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import dataUtils
# plotUtils 和 tensorflow_addons 暂时在代码中没有使用，如果需要可以导入
# import plotUtils
import tensorflow_addons as tfa

# 定义加载模型的函数
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# 定义测试函数
def test_model(model, test_dataset):
    # 初始化评估指标
    accuracy_metric = tf.keras.metrics.Accuracy()

    # 初始化计数器
    total_samples = 0
    correct_predictions = 0

    # 逐批处理测试数据
    for batch_index, (batch_x, batch_y) in enumerate(test_dataset):
        # 使用模型进行预测
        predictions = model.predict_on_batch(batch_x)
        predicted_classes = np.argmax(predictions, axis=1)

        # 更新准确率指标
        accuracy_metric.update_state(batch_y, predicted_classes)

        # 更新计数器
        correct_predictions += np.sum(predicted_classes == batch_y)
        total_samples += batch_y.shape[0]

        # 打印当前批次的准确率信息
        batch_accuracy = np.mean(predicted_classes == batch_y)
        print(f'Batch {batch_index + 1}: Batch Accuracy: {batch_accuracy * 100:.2f}%, Total Tested: {total_samples}')

    # 获取最终的准确率结果
    accuracy = accuracy_metric.result().numpy()
    return accuracy, total_samples, correct_predictions

# 定义主函数
def main(test_path, model_path, batch_size=32):
    # 加载测试数据
    test_x, test_y, _, _ = dataUtils.get_data(test_path, 'position', train_ratio=0.9, data_augmentation=False)

    # 创建测试数据集对象
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)

    # 加载模型
    model = load_trained_model(model_path)

    # 测试模型
    accuracy, total_samples, correct_predictions = test_model(model, test_dataset)

    # 打印最终结果
    print(f'Finished testing {total_samples} samples.')
    print(f'Correct Predictions: {correct_predictions}')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 如果这是主程序，则运行主函数
if __name__ == '__main__':
    test_path = "train.csv"
    model_path = 'Data/model.h5'
    main(test_path, model_path)
