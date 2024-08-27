import tensorflow as tf
import numpy as np

# 加载模型
model_path = r'D:\repository\deeplearning\small\Data\tensorflow\ncz\1'
model = tf.saved_model.load(model_path)
print("模型加载成功")

# 定义姿势标签
posture_labels = ['平躺', '左侧卧', '右侧卧']

def predict_posture(matrix):
    matrix = np.array(matrix, dtype=np.float32)
    print(matrix.shape)
    
    # 处理输入数据
    input_data = matrix.flatten().reshape(1, 160, 1)
    
    input_data = tf.cast(input_data, tf.float32)
    
    # 进行预测
    predictions = model(input_data, training=False)
    
    # 获取预测结果
    predicted_class = tf.argmax(predictions[0]).numpy()
    confidence = tf.reduce_max(predictions[0]).numpy()
    
    # 获取所有类别的概率
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    
    return predicted_class, confidence, probabilities

# 主函数
def main():
    sample_matrix = [0,0,1,22,8,1,0,0,0,0,0,0,0,6,4,1,0,0,0,0,0,3,27,28,36,50,56,43,29,2,1,20,45,39,37,38,9,1,0,0,2,36,7,40,48,33,3,0,0,0,2,1,2,21,33,24,2,0,0,0,0,0,1,13,26,19,4,0,0,0,0,1,13,37,34,70,51,3,0,0,1,2,13,35,52,57,39,5,0,0,0,1,14,23,4,6,30,31,1,0,0,0,1,14,1,1,5,20,0,0,0,0,0,1,0,0,0,2,1,0,0,0,0,3,0,1,0,1,33,5,0,0,2,29,3,1,0,1,19,29,0,0,1,13,1,0,0,0,2,32,1,2,4,5,1,2,1,1,2,7]
    
    # sample_matrix = sample_matrix.astype(np.float32)
    
    predicted_class, confidence, probabilities = predict_posture(sample_matrix)
    
    print("预测结果:")
    print(f"姿势: {posture_labels[predicted_class]}")
    print(f"置信度: {confidence:.2f}")
    print("\n各类别概率:")
    for i, prob in enumerate(probabilities):
        print(f"{posture_labels[i]}: {prob:.4f}")

if __name__ == "__main__":
    main()