# ./models/mnist_cnn.h5

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisasi ke [0,1] dan reshape ke (28,28,1)
    x_train = (x_train.astype('float32') / 255.0)[..., None]
    x_test = (x_test.astype('float32') / 255.0)[..., None]

    # Split a validation set manually (needed because datagen.flow() doesn't support validation_split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    # Data augmentation so the model generalizes to hand-drawn canvas digits,
    # not just clean, pre-centered MNIST scans
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=8
    )
    datagen.fit(x_train)

    model = build_model()
    model.summary()

    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        validation_data=(x_val, y_val),
        epochs=10,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/mnist_cnn.h5'
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()