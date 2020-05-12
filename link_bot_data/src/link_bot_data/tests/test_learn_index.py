import tensorflow as tf
import numpy as np
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)

indices = tf.data.Dataset.range(8)
inputs = indices.map(lambda index: tf.one_hot(index, depth=8))
dataset = tf.data.Dataset.zip((inputs, indices)).repeat(1024).shuffle(seed=1, buffer_size=1024).batch(32)

model = tf.keras.Sequential([
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(loss=tf.keras.losses.mse)
model.fit(x=dataset, epochs=50)


def f(index):
    x = np.zeros(8)
    x[index] = 1
    return x


for i in range(8):
    print(i, model.predict(tf.expand_dims(f(i), axis=0)))

print(model.summary())
model.weights
