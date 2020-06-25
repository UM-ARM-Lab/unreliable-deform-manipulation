import tensorflow as tf
from moonshine.gpu_config import limit_gpu_mem


limit_gpu_mem(1.0)

l = tf.keras.layers.Conv2D(2, [3, 3])


@tf.function
def f(time):
    xs = []
    a = tf.random.uniform([5, 10, 10, 3])
    for i in range(time):
        b = l(a)
        xs.append(b)
    return tf.stack(xs, axis=1)


my_xs = f(3)
print(my_xs.shape)
