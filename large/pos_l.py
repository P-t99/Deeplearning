# 导入Python未来版本的特性，这里特指print函数的特性
from __future__ import print_function

# 导入tensorflow的keras模块，一个用于构建和训练深度学习模型的高级API
import tensorflow.keras as keras

# 导入keras的多个网络层，这些层将用于构建神经网络模型
from tensorflow.keras.layers import Input, Dense, Flatten, add
from tensorflow.keras.layers import Conv2D, Activation, AveragePooling2D, MaxPooling2D

# 导入Keras后端，这是一个抽象层，使得你可以编写同时适用于多种深度学习库的代码
from tensorflow.keras import backend as K

# 导入tensorflow库，一个用于机器学习和深度神经网络的库
import tensorflow as tf

# 导入Model类，用于实例化一个训练和预测功能的模型对象
from tensorflow.keras.models import Model

# 导入自定义的数据处理和绘图工具模块
import dataUtils, plotUtils

# 导入TensorFlow Addons，一个包含了额外层、损失函数、优化器等工具的库
import tensorflow_addons as tfa

# 配置TensorFlow以允许显存动态增长，避免一次性占用全部显存
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# 创建一个新的TensorFlow会话，并应用上面的配置
sess = tf.compat.v1.Session(config=config)


# 定义一个残差块函数，用于构建残差网络结构
def res_block(x, channels, i):
    # 如果是第一个残差块，步长为1，否则为2，并添加卷积层
    if i == 1:
        strides = (1, 1)
        x_add = x  # 第一个块直接使用输入x作为快捷连接
    else:
        strides = (2, 2)
        x_add = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same', strides=strides)(
            x)  # 创建一个卷积层作为快捷连接

    # 添加两个卷积层，第一个使用relu激活函数，第二个不使用
    x = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(channels, kernel_size=(3, 3), padding='same', strides=strides)(x)
    # 将输入和卷积结果相加
    x = add([x, x_add])
    # 应用relu激活函数
    x = Activation(K.relu)(x)
    # 返回残差块的输出
    return x


# 定义模型构建函数
def build_model(input_shape, num_classes):
    # 创建模型输入层
    inpt = Input(shape=input_shape)
    # 对输入数据进行预处理并转换维度形状
    x = keras.layers.Lambda(lambda y: tf.reshape(y, [-1, 32, 32, 1]))(inpt)
    # 对输入数据应用高斯滤波
    x = keras.layers.Lambda(lambda y: tfa.image.gaussian_filter2d(y, [9, 9], 1, padding='SYMMETRIC'))(x)
    # 增加图像的锐度
    x = keras.layers.Lambda(lambda y: tfa.image.sharpness(y, 0.15))(x)
    # 对图像应用逐个图像的标准化
    x = keras.layers.Lambda(lambda y: tf.image.per_image_standardization(y))(x)
    # 在模型中添加高斯噪声层
    x = keras.layers.GaussianNoise(0.2)(x)
    # 添加第一个卷积层
    x = Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=input_shape, padding='same')(x)
    # 下面是调用自定义的残差块构建复杂的网络结构
    for i in range(2):
        x = res_block(x, 16, i)
    for i in range(2):
        x = res_block(x, 32, i)
    # 应用平均池化层
    x = AveragePooling2D(pool_size=(7, 7))(x)
    # 对特征图进行展平操作
    x = Flatten()(x)
    # 添加全连接网络层
    for n in range(3):
        x = Dense(32, activation='relu')(x)
        # 在全连接网络层后面添加dropout层以减少过拟合
        x = keras.layers.Dropout(0.5)(x)
    # 添加输出层
    x = Dense(num_classes, activation='softmax')(x)
    # 创建模型对象
    model = Model(inputs=inpt, outputs=x)
    # 定义模型的优化器
    OP = keras.optimizers.Adam(lr=0.00015, epsilon=None, decay=0.0)
    # 编译模型，设置损失函数和评估指标
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=OP,
                  metrics=['acc'])

    # 返回构建好的模型
    return model


# 定义主函数，包括训练、评估和保存模型的操作
def main(train_path, test_path, outpath, epochs, batch_size, num_classes):
    # 将字符串参数转换为整数
    epochs = int(epochs)
    batch_size = int(batch_size)
    # 设置要预测的标签
    label = "position"
    # 使用自定义的dataUtils来载入训练和测试数据
    train_x, train_y, test_x, test_y = dataUtils.get_data(train_path, label, train_ratio=0.9, data_augmentation='False')
    # 打印加载的数据和测试数据的长度
    print(train_x, len(test_x))
    # 遍历每个训练样本，并逐个打印出来
    for index in range(len(train_x)):
        print(train_x[index])
        print(train_y[index])

    # 调用build_model函数来创建模型
    model = build_model([1024, 1], num_classes)
    # 设置学习率调整策略
    dynamic_LR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                   patience=10, verbose=0, mode='auto',
                                                   epsilon=0.0001, cooldown=0, min_lr=0)
    # 设置提前终止训练的策略
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='auto')
    # 设置模型保存策略，只保存效果最好的模型
    checkpoint_cb = keras.callbacks.ModelCheckpoint(outpath + "/model.h5", save_best_only=True)

    # 训练模型
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[dynamic_LR, early_stopping, checkpoint_cb],
                        validation_split=0.3)

    # 评估模型
    loss1, acc1 = model.evaluate(train_x, train_y)
    # 打印模型准确率
    print('Res test_2,acc:{:5.2f}%'.format(100 * acc1))

    # 使用自定义工具对训练过程的历史数据进行绘图
    plotUtils.plot_history(history, outpath)
    # 绘制混淆矩阵
    plotUtils.plot_confusion_matrix(model, test_x, test_y, outpath)
    # 保存模型
    tf.saved_model.save(model, outpath + '/tensorflow/ncz/1')



# 如果这个脚本作为主脚本执行，则执行main函数
if __name__ == '__main__':
    # 定义训练数据路径
    train_path = "sdata.csv"
    # 定义测试数据路径
    test_path = "sdata.csv"
    # 定义模型输出路径
    outpath = 'Data'
    # 定义每批次训练的样本数量
    batch_size = 50
    # 定义分类类别数
    num_classes = 3
    # 定义训练时代数
    epochs = 50
    # 调用main函数
    main(train_path, test_path, outpath, epochs, batch_size, num_classes)