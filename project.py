import pandas as pd
import numpy as np
import tensorflow as tf
import model
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_score, recall_score

mnist = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)
X, y = mnist[0], mnist[1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train_int = y_train.astype('int')
y_test_int = y_test.astype('int')

num_classes = 10
y_train_one_hot = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes)

cnn_1layer = model.cnn_model1()
cnn_3layer = model.cnn_model2()

cnn_1layer.fit(x_train, y_train_one_hot, epochs=20, verbose=True)
cnn_3layer.fit(x_train, y_train_one_hot, epochs=20, verbose=True)

models = [('CNN 1 Layer', cnn_1layer), ('CNN 3 Layers', cnn_3layer)]

for name, model in models:
    results = model.evaluate(x_test, y_test_one_hot, verbose=0)
    print(f"\nModel: {name}")
    for metric_name, result in zip(model.metrics_names, results):
        print(f"{metric_name}: {result:.4f}")

y_pred = cnn_1layer.predict(x_test)
y_pred2 = cnn_3layer.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("Precision (1-layer):", precision_score(y_test_int, y_pred_labels, average='macro'))
print("Recall (1-layer):", recall_score(y_test_int, y_pred_labels, average='macro'))

y_pred_labels2 = np.argmax(y_pred2, axis=1)
print("Precision (3-layer):", precision_score(y_test_int, y_pred_labels2, average='macro'))
print("Recall (3-layer):", recall_score(y_test_int, y_pred_labels2, average='macro'))
