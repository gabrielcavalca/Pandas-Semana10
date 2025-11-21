import tensorflow as tf
from tensorflow import keras

def cnn_model1():

    deepModel = keras.Sequential([

        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            input_shape=(28, 28, 1)
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(
            64,
            activation='relu',
            kernel_initializer='he_normal'
        ),

        keras.layers.Dense(10, activation='softmax')
    ])

    deepModel.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision', top_k=1),
            keras.metrics.Recall(name='recall', top_k=1)
        ]
    )

    return deepModel

def cnn_model2():

    deepModel = keras.Sequential([

        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal', input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(10, activation='softmax')
    ])

    deepModel.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision', top_k=1),
            keras.metrics.Recall(name='recall', top_k=1)
        ]
    )

    return deepModel
