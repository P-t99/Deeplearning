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

# 设置GPU配置，允许显存动态增长，避免一次性占用全部显存
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# 定义一个残差块函数，用于构建残差网络结构
def res_block(x, channels, i):
    # 如果是第一个残差块，步长为1，否则为2，并添加卷积层
    if i == 1:
        strides = (1, 1)
        x_add = x
    else:
        strides = (2, 2)
        x_add = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same', strides=strides)(x)

    # 添加两个卷积层，第一个使用relu激活函数，第二个不使用
    x = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(channels, kernel_size=(3, 3), padding='same', strides=strides)(x)
    # 将输入和卷积结果相加
    x = add([x, x_add])
    # 应用激活函数并返回残差块的输出
    x = Activation(K.relu)(x)
    return x

def build_model(input_shape, num_classes):

    inpt = Input(shape=input_shape)
    x = keras.layers.Lambda(lambda y: tf.reshape(y, [-1, 16, 10, 1]))(inpt)
    x = keras.layers.Lambda(lambda y: tfa.image.gaussian_filter2d(y, [3, 3], 1, padding='SYMMETRIC'))(x)
    x = keras.layers.Lambda(lambda y: tfa.image.sharpness(y, 0.15))(x)
    x = keras.layers.Lambda(lambda y: tf.image.per_image_standardization(y))(x)
    x = keras.layers.GaussianNoise(0.2)(x)

    x = Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same')(x)

    for i in range(2):
        x = res_block(x, 16, i)

    for i in range(2):
        x = res_block(x, 32, i)

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    for n in range(3):
        x = Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)

    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    OP = keras.optimizers.Adam(lr=0.00015)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=OP,
                  metrics=['acc'])
    return model

def main(train_path, test_path, outpath, epochs, batch_size, num_classes):
    epochs = int(epochs)
    batch_size = int(batch_size)
    label = "position"
    train_x, train_y, test_x, test_y = dataUtils.get_data(train_path, label, train_ratio=0.9, data_augmentation='False')
    
    # 打印少量样本数据
    for index in range(min(len(train_x), 5)):
        print(train_x[index])
        print(train_y[index])

    model = build_model([160, 1], num_classes)
    
    dynamic_LR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                   patience=10, verbose=0, mode='auto',
                                                   cooldown=0, min_lr=0)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='auto')
    checkpoint_cb = keras.callbacks.ModelCheckpoint(outpath+"/model.h5", save_best_only=True)

    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[dynamic_LR, early_stopping, checkpoint_cb],
                        validation_split=0.3)

    loss1, acc1 = model.evaluate(train_x, train_y)
    print('Res test_2,acc:{:5.2f}%'.format(100*acc1))

    plotUtils.plot_history(history, outpath)
    tf.saved_model.save(model, outpath+'/tensorflow/ncz/1')
    plotUtils.plot_confusion_matrix(model, test_x, test_y, outpath)

if __name__ == '__main__':
    train_path = "train.csv"
    test_path = "train.csv"
    outpath = 'Data'
    batch_size = 50
    num_classes = 3
    epochs = 200
    main(train_path, test_path, outpath, epochs, batch_size, num_classes)
