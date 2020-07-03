import unittest
import tensorflow as tf
from moonshine.gpu_config import limit_gpu_mem


limit_gpu_mem(1.0)

l = tf.keras.layers.Conv2D(2, [3, 3])


@tf.function
def int_f(time):
    xs = []
    a = tf.random.uniform([5, 10, 10, 3])
    print(type(time))
    tf.print(type(time))
    for i in range(time):
        b = l(a)
        xs.append(b)
    return tf.stack(xs, axis=1)


@tf.function
def tf_f(time):
    xs = []
    a = tf.random.uniform([5, 10, 10, 3])
    time = tf.constant(time)
    tf.print(type(time))
    print(type(time))
    for i in range(time):
        b = l(a)
        xs.append(b)
    return tf.stack(xs, axis=1)


class TestLoop(unittest.TestCase):

    def test_int_loop(self):
        my_xs = int_f(3)
        self.assertEqual(my_xs.shape, [5, 3, 8, 8, 2])

    def test_tf_loop(self):
        with self.assertRaises(Exception):
            tf_f(3)

    def test_tensor_array(self):
        x = tf.TensorArray(tf.float32, 10, dynamic_size=True)
        for i in tf.range(10):


if __name__ == "__main__":
    unittest.main()
