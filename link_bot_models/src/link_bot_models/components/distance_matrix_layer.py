import tensorflow as tf
from keras.layers import Layer


class DistanceMatrix(Layer):

    def __init__(self, **kwargs):
        super(DistanceMatrix, self).__init__(**kwargs)
        self.n = None
        self.n_points = None

    def build(self, input_shape):
        super(DistanceMatrix, self).build(input_shape)  # Be sure to call this at the end
        self.n = input_shape[1]
        self.n_points = int(input_shape[1] // 2)

    def call(self, x, **kwargs):
        x_as_points = tf.reshape(x, [-1, self.n_points, 2])
        # x is a ?xNx2 tensor
        x_matrix = tf.reshape(tf.tile(x_as_points, [1, self.n_points, 1], name='x_matrix'), [-1, self.n_points, self.n_points, 2])
        # x_matrix is ?xNxNx2
        distances = tf.norm(x_matrix - tf.transpose(x_matrix, [0, 2, 1, 3], name='x_matrix_T'), axis=3, keepdims=True)
        # distances is ?xNxN
        return distances

    def get_config(self):
        config = {}
        base_config = super(DistanceMatrix, self).get_config()
        return base_config.update(config)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_points, self.n_points, 1
