from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from dqn import QNet

x = np.random.randn(10000,7)
y = np.random.randint(4, size=10000)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(7, ), activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_hat = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(7, ), activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, epochs=100)
x_pred = np.random.randn(16,7)
preds = np.argmax(model.predict(x_pred), axis=1)
preds_hat = np.argmax(model_hat.predict(x_pred), axis=1)
