import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

EPOCHS = int(os.environ.get('EPOCHS', 15))
BATCH = 128
SEED = 42
MODEL_PATH = 'models/mnist_cnn.keras'


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
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')[..., None] / 255.0
    x_test = x_test.astype('float32')[..., None] / 255.0

    # hold out 10% of the training images for validation
    idx = np.random.RandomState(SEED).permutation(len(x_train))
    n_val = len(x_train) // 10
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return ((x_train[train_idx], y_train[train_idx]),
            (x_train[val_idx], y_train[val_idx]),
            (x_test, y_test))


#  augmentation 
geometry = tf.keras.Sequential([
    layers.RandomRotation(10 / 360, fill_mode='constant', fill_value=0.0),
    layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0),
    layers.RandomZoom(0.1, fill_mode='constant', fill_value=0.0),
])


def dilate(x):   # thicker strokes
    return tf.nn.max_pool2d(x, 3, 1, 'SAME')


def erode(x):    # thinner strokes
    return -tf.nn.max_pool2d(-x, 3, 1, 'SAME')


def stroke_width_aug(x):
    """Per image: ~55% unchanged, 20% slightly thicker, 10% much thicker, 15% thinner."""
    r = tf.random.uniform([tf.shape(x)[0], 1, 1, 1])
    thick1 = dilate(x)
    thick2 = dilate(thick1)
    out = tf.where(r < 0.20, thick1, x)
    out = tf.where((r >= 0.20) & (r < 0.30), thick2, out)
    out = tf.where((r >= 0.30) & (r < 0.45), erode(x), out)
    return out


def make_train_ds(x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=SEED).batch(BATCH)

    def augment(xb, yb):
        xb = geometry(xb, training=True)
        xb = stroke_width_aug(xb)
        return tf.clip_by_value(xb, 0.0, 1.0), yb

    return ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def main():
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = get_data()

    model = build_model()
    model.summary()

    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, restore_best_weights=True)
    model.fit(make_train_ds(x_tr, y_tr), validation_data=(x_val, y_val),
              epochs=EPOCHS, callbacks=[stop], shuffle=False, verbose=2)

    _, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f'Test accuracy: {acc:.4f}')

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')


if __name__ == '__main__':
    main()