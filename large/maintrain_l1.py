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
from sklearn.model_selection import StratifiedKFold
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

def get_data(filepath, label, normalization=False, shuffle=True, test_size=0.2, data_augmentation='False'):
    x, y = get_raw_data_from_csv(filepath, label, normalization, data_augmentation)

    if shuffle:
        permutation = np.random.permutation(y.shape[0])
        x = x[permutation]
        y = y[permutation]

    train_total = x.shape[0]
    test_nums = int(train_total * test_size)

    x_train = x[:-test_nums]
    y_train = y[:-test_nums]
    x_test = x[-test_nums:]
    y_test = y[-test_nums:]

    return x_train, y_train, x_test, y_test

# Optimized KeepTop32 layer
import tensorflow as tf

class KeepTop32(tf.keras.layers.Layer):
    def __init__(self, k=400, **kwargs):
        super(KeepTop32, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        # 输入形状应为 (batch_size, 16, 10, 1)
        
        # 展平空间维度和通道维度
        flattened = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        
        # 获取 top k 个值
        values, _ = tf.nn.top_k(flattened, k=self.k)
        
        # 获取阈值（第k个最大值）
        thresholds = values[:, -1]
        
        # 重塑阈值以便广播
        thresholds = tf.reshape(thresholds, [-1, 1, 1, 1])
        
        # 创建并应用掩码
        mask = tf.cast(tf.greater_equal(inputs, thresholds), inputs.dtype)
        return inputs * mask

    def get_config(self):
        config = super(KeepTop32, self).get_config()
        config.update({"k": self.k})
        return config

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
    
    x = Lambda(lambda y: tf.reshape(y, [-1, 32, 32, 1]))(inpt)
    x = KeepTop32()(x)
    x = Lambda(lambda y: tfa.image.gaussian_filter2d(y, [3, 3], 1, padding='SYMMETRIC'))(x)
    x = Lambda(lambda y: tfa.image.sharpness(y, 0.15))(x)
    x = Lambda(lambda y: tf.image.per_image_standardization(y))(x)
    
    x = Lambda(lambda y: y + tf.random.normal(tf.shape(y), mean=0.0, stddev=0.2) * tf.cast(tf.keras.backend.learning_phase(), tf.float32))(x)

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)

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

# Plotting functions
def plot_history(histories, path):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['acc'], label=f'Training acc (Fold {i+1})')
        plt.plot(history.history['val_acc'], label=f'Validation acc (Fold {i+1})')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Training loss (Fold {i+1})')
        plt.plot(history.history['val_loss'], label=f'Validation loss (Fold {i+1})')
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
def main(train_path, test_path, outpath, epochs, batch_size, num_classes, n_splits=5):
    epochs = int(epochs)
    batch_size = int(batch_size)
    label = "position"
    x_train, y_train, x_test, y_test = get_data(train_path, label, test_size=0.2, data_augmentation='False')
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    histories = []
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(x_train, y_train), 1):
        print(f"\nFold {fold}")
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = build_model([1024, 1], num_classes)
        
        if fold == 1:
            model.summary()

        OP = keras.optimizers.Adam(learning_rate=0.00015)
        model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                      optimizer=OP,
                      metrics=['acc'])
        
        dynamic_LR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                       patience=10, verbose=0, mode='auto',
                                                       cooldown=0, min_lr=0)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='auto')
        checkpoint_cb = keras.callbacks.ModelCheckpoint(outpath+f"/model_fold{fold}.h5", save_best_only=True)

        history = model.fit(x_train_fold, y_train_fold,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[dynamic_LR, early_stopping, checkpoint_cb],
                            validation_data=(x_val_fold, y_val_fold))

        histories.append(history)
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(x_val_fold, y_val_fold, verbose=0)
        print(f'Validation accuracy (Fold {fold}): {val_acc:.4f}')
        fold_accuracies.append(val_acc)

    print("\nCross-validation results:")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"Fold {i}: {acc:.4f}")
    print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

    plot_history(histories, outpath)
    
    # Evaluate on test set using the model from the best fold
    best_fold = np.argmax(fold_accuracies) + 1
    best_model = keras.models.load_model(outpath+f"/model_fold{best_fold}.h5", custom_objects={'KeepTop32': KeepTop32})
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest accuracy (using best model from fold {best_fold}): {test_acc:.4f}')

    # Save the best model
    tf.saved_model.save(best_model, outpath+'/tensorflow/ncz/1')
    
    plot_confusion_matrix(best_model, x_test, y_test, outpath)

if __name__ == '__main__':
    train_path = "train.csv"
    test_path = "train.csv"
    outpath = 'Data'
    batch_size = 64
    num_classes = 3
    epochs = 10
    n_splits = 5  # 5-fold cross-validation
    main(train_path, test_path, outpath, epochs, batch_size, num_classes, n_splits)
    
    
    