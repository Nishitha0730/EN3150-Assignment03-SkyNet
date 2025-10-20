# custom_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from utils import load_mnist_flat, plot_history, evaluate_and_report

def build_simple_cnn(input_shape=(28,28,1), num_classes=10, dropout_rate=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_compare_optimizers():
    X_train, X_val, X_test, y_train_o, y_val_o, y_test_o, y_train_raw, y_val_raw, y_test_raw = load_mnist_flat()
    input_shape = X_train.shape[1:]

    # We'll compare Adam, SGD, SGD+momentum
    optimizers_to_try = {
        'Adam': optimizers.Adam(learning_rate=1e-3),
        'SGD': optimizers.SGD(learning_rate=1e-2),
        'SGD_momentum': optimizers.SGD(learning_rate=1e-2, momentum=0.9)
    }

    results = {}
    for name, opt in optimizers_to_try.items():
        print(f"\n=== Training with optimizer: {name} ===")
        model = build_simple_cnn(input_shape=input_shape)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(
            X_train, y_train_o,
            epochs=20,
            batch_size=128,
            validation_data=(X_val, y_val_o),
            verbose=2
        )
        plot_history(history, title_prefix=f'CustomCNN - {name}')
        print(f"Evaluation for optimizer {name}:")
        evaluate_and_report(model, X_test, y_test_o, y_test_raw)
        results[name] = history
    return results

if __name__ == "__main__":
    train_and_compare_optimizers()
