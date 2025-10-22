# utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_mnist_flat():
    """Load MNIST from Keras and normalize. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    # normalize and expand channel dim
    X = X.astype("float32") / 255.0
    X = np.expand_dims(X, -1)  # shape (N, 28, 28, 1)
    # initial split 70/30, then split 30 into 15/15 (val/test)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, train_size=0.5, stratify=y_rest, random_state=42)
    # one-hot encode labels
    y_train_o = to_categorical(y_train, num_classes=10)
    y_val_o = to_categorical(y_val, num_classes=10)
    y_test_o = to_categorical(y_test, num_classes=10)
    return X_train, X_val, X_test, y_train_o, y_val_o, y_test_o, y_train, y_val, y_test

def plot_history(history, title_prefix=""):
    """Plot training & validation loss and accuracy."""
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history.get('val_loss', []), label='val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title(f'{title_prefix} Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history.get('accuracy', history.history.get('acc')), label='train acc')
    plt.plot(history.history.get('val_accuracy', history.history.get('val_acc')), label='val acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title(f'{title_prefix} Accuracy')
    plt.tight_layout()
    plt.show()

def evaluate_and_report(model, X_test, y_test_o, y_test_raw, class_names=None):
    """Evaluate model and print_metrics + confusion matrix."""
    loss, acc = model.evaluate(X_test, y_test_o, verbose=0)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)

    print("\nClassification report:")
    print(classification_report(y_test_raw, y_pred, digits=4))

    cm = confusion_matrix(y_test_raw, y_pred)
    plot_confusion_matrix(cm, classes=class_names or list(range(cm.shape[0])))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(7,6))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
