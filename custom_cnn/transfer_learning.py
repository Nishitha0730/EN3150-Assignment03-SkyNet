# transfer_learning.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from utils import load_mnist_flat, plot_history, evaluate_and_report
import numpy as np

def preprocess_for_vgg(X):
    # VGG expects 3 channels and at least 48x48; we will resize to 48x48 and convert to 3 channels
    X_resized = tf.image.resize(X, [48,48]).numpy()
    X_rgb = np.concatenate([X_resized, X_resized, X_resized], axis=-1)
    # Use imagenet preprocessing
    X_rgb = applications.vgg16.preprocess_input(X_rgb * 255.0)
    return X_rgb

def build_finetune_vgg(input_shape=(48,48,3), num_classes=10, dropout_rate=0.5):
    base = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False  # freeze base for initial training
    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(base.input, out)
    return model, base

def train_finetune_vgg():
    X_train, X_val, X_test, y_train_o, y_val_o, y_test_o, y_train_raw, y_val_raw, y_test_raw = load_mnist_flat()

    X_train_p = preprocess_for_vgg(X_train)
    X_val_p = preprocess_for_vgg(X_val)
    X_test_p = preprocess_for_vgg(X_test)

    model, base = build_finetune_vgg(input_shape=X_train_p.shape[1:])
    model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Initial training with frozen base
    history1 = model.fit(X_train_p, y_train_o, validation_data=(X_val_p, y_val_o),
                         epochs=10, batch_size=64, verbose=2)

    # Unfreeze some top layers and fine-tune
    for layer in base.layers[-4:]:
        layer.trainable = True

    model.compile(optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history2 = model.fit(X_train_p, y_train_o, validation_data=(X_val_p, y_val_o),
                         epochs=10, batch_size=64, verbose=2)

    # Combine histories (for plotting)
    class HistoryCombined:
        def __init__(self, h1, h2):
            self.history = {}
            for k in set(h1.history.keys()).union(h2.history.keys()):
                self.history[k] = h1.history.get(k, []) + h2.history.get(k, [])

    combined = HistoryCombined(history1, history2)
    plot_history(combined, title_prefix='VGG16 Fine-tune')

    evaluate_and_report(model, X_test_p, y_test_o, y_test_raw)

if __name__ == "__main__":
    train_finetune_vgg()
