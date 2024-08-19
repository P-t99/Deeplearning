from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Flatten, add, Conv2D, Activation, AveragePooling2D, MaxPooling2D, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
import dataUtils, plotUtils
import tensorflow_addons as tfa

# 设置GPU配置
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def res_block(x, channels, i):
    if i == 1:
        strides = (1, 1)
        x_add = x
    else:
        strides = (2, 2)
        x_add = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same', strides=strides)(x)

    x = Conv2D(channels, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(channels, kernel_size=(3, 3), padding='same', strides=strides)(x)
    x = add([x, x_add])
    x = Activation(K.relu)(x)
    return x

def build_model(input_shape, num_classes):
    inpt = Input(shape=input_shape)
    
    # 集成预处理步骤
    x = Lambda(lambda y: tf.reshape(y, [-1, 16, 10, 1]))(inpt)
    x = Lambda(lambda y: tfa.image.gaussian_filter2d(y, [3, 3], 1, padding='SYMMETRIC'))(x)
    x = Lambda(lambda y: tfa.image.sharpness(y, 0.15))(x)
    x = Lambda(lambda y: tf.image.per_image_standardization(y))(x)
    
    # 高斯噪声只在训练时使用
    x = Lambda(lambda y: y + tf.random.normal(tf.shape(y), mean=0.0, stddev=0.2) * tf.cast(tf.keras.backend.learning_phase(), tf.float32))(x)

    x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x)

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
    
    OP = keras.optimizers.Adam(lr=0.00015)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=OP,
                  metrics=['acc'])
    
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
    
    # 保存完整模型，包括预处理步骤
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