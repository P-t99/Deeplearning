import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_history(history, path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.savefig(path+"/Training_and_validation_accuracy.png")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig(path + "/Training_and_validation_loss.png")
    plt.legend()
    plt.show()

def plotmatrix(path, cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
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

