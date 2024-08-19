import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Flatten, add, Conv2D, Activation, AveragePooling2D, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools

# GPU configuration
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# Data processing functions
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

# New function to keep top 32 points
def keep_top_32(x):
    values, _ = tf.nn.top_k(tf.reshape(x, [-1, 160]), k=32)
    min_value = values[:, -1:]
    mask = tf.cast(tf.greater_equal(x, min_value), tf.float32)
    return x * mask

# Model building functions
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
    
    # Integrated preprocessing steps
    x = Lambda(keep_top_32)(inpt)  # New step to keep top 32 points
    x = Lambda(lambda y: tf.reshape(y, [-1, 16, 10, 1]))(x)
    x = Lambda(lambda y: tfa.image.gaussian_filter2d(y, [3, 3], 1, padding='SYMMETRIC'))(x)
    x = Lambda(lambda y: tfa.image.sharpness(y, 0.15))(x)
    x = Lambda(lambda y: tf.image.per_image_standardization(y))(x)
    
    # Gaussian noise only during training
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

# Updated plotting function
def plot_history(history, path):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'], label='Training acc')
    plt.plot(history.history['val_acc'], label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path + "/Training_and_validation_metrics.png")
    plt.show()

def plotmatrix(path, cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path + "/Confusion_matrix.png")
    plt.show()

def plot_confusion_matrix(model, x_test, y_test, path):
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    matrix = metrics.confusion_matrix(y_test, y_pred)
    plt.figure()
    plotmatrix(path, matrix, range(np.max(y_test)+1))

# Main function
def main(train_path, test_path, outpath, epochs, batch_size, num_classes):
    epochs = int(epochs)
    batch_size = int(batch_size)
    label = "position"
    train_x, train_y, test_x, test_y = get_data(train_path, label, train_ratio=0.9, data_augmentation='False')
    
    # Print a few sample data points
    for index in range(min(len(train_x), 5)):
        print(train_x[index])
        print(train_y[index])

    model = build_model([160, 1], num_classes)
    
    OP = keras.optimizers.Adam(learning_rate=0.00015)
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

    plot_history(history, outpath)
    
    # Save the complete model, including preprocessing steps
    tf.saved_model.save(model, outpath+'/tensorflow/ncz/1')
    
    plot_confusion_matrix(model, test_x, test_y, outpath)

if __name__ == '__main__':
    train_path = "train.csv"
    test_path = "train.csv"
    outpath = 'Data'
    batch_size = 50
    num_classes = 3
    epochs = 200
    main(train_path, test_path, outpath, epochs, batch_size, num_classes)