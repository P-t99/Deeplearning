import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa

class KeepTopN(layers.Layer):
    def __init__(self, n=48, **kwargs):
        super(KeepTopN, self).__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        # 输入形状为 (batch_size, 16, 10, 1)
        
        # 展平空间维度和通道维度
        flattened = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        
        # 获取 top n 个值
        values, _ = tf.nn.top_k(flattened, k=self.n)
        
        # 获取阈值（第n个最大值）
        thresholds = values[:, -1]
        
        # 重塑阈值以便广播
        thresholds = tf.reshape(thresholds, [-1, 1, 1, 1])
        
        # 创建并应用掩码
        mask = tf.cast(tf.greater_equal(inputs, thresholds), inputs.dtype)
        return inputs * mask

    def get_config(self):
        config = super(KeepTopN, self).get_config()
        config.update({"n": self.n})
        return config

def res_block(x, channels, i):
    if i == 1:
        strides = (1, 1)
        x_add = x
    else:
        strides = (2, 2)
        x_add = layers.Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same', strides=strides)(x)

    x = layers.Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(channels, kernel_size=(3, 3), padding='same', strides=strides)(x)
    x = layers.Add()([x, x_add])
    x = layers.Activation('relu')(x)
    return x

def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # 重塑为 2D 图像
    x = layers.Lambda(lambda y: tf.reshape(y, [-1, 16, 10, 1]))(inputs)
    
    # 应用 KeepTopN 层
    x = KeepTopN(n=48)(x)
    
    # 添加图像处理步骤
    x = layers.Lambda(lambda y: tfa.image.gaussian_filter2d(y, [3, 3], 1, padding='SYMMETRIC'))(x)
    x = layers.Lambda(lambda y: tfa.image.sharpness(y, 0.15))(x)
    x = layers.Lambda(lambda y: tf.image.per_image_standardization(y))(x)
    
    # 添加噪声（仅在训练时）
    x = layers.Lambda(lambda y: y + tf.random.normal(tf.shape(y), mean=0.0, stddev=0.2) * 
                      tf.cast(tf.keras.backend.learning_phase(), tf.float32))(x)

    # 初始卷积层
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # 添加残差块
    for i in range(2):
        x = res_block(x, 16, i)

    for i in range(2):
        x = res_block(x, 32, i)

    # 平均池化
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    # 展平
    x = layers.Flatten()(x)
    
    # 全连接层
    for _ in range(3):
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

    # 输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# 使用示例
if __name__ == "__main__":
    # 定义输入形状和类别数量
    input_shape = (160, 1)  # 输入形状，不包括批量大小
    num_classes = 4  # 假设有4个分类
    
    # 构建模型
    model = build_model(input_shape, num_classes)
    
    # 打印模型摘要
    model.summary()

    # 可选：保存模型结构图
    # tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True)