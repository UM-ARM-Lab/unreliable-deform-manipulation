from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Lambda, Concatenate, Reshape, Activation
from keras.models import Model

from link_bot_models.base_model import BaseModel
from link_bot_models.layers.tf_signed_distance_field_op import SDFLookup
from link_bot_models.layers.bias_layer import BiasLayer
from link_bot_pycommon import link_bot_pycommon


class SDFFuncationModel(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(SDFFuncationModel, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        # we have to flatten everything in order to pass it around and I don't understand why
        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        self.fc_layer_sizes = [
            16,
            16,
        ]

        self.beta = 1e-2

        def oob_regularization(sdf_input_points):
            # FIXME: this assumes that the physical world coordinates (0,0) in meters is the origin/center of the SDF
            distances_to_origin = tf.norm(sdf_input_points, axis=1)
            oob_left = sdf_input_points[:, 0] <= sdf_extent[:, 0]
            oob_right = sdf_input_points[:, 0] >= sdf_extent[:, 1]
            oob_up = sdf_input_points[:, 1] <= sdf_extent[:, 2]
            oob_down = sdf_input_points[:, 1] >= sdf_extent[:, 3]
            out_of_bounds = tf.math.reduce_any(tf.stack((oob_up, oob_down, oob_left, oob_right), axis=1), axis=1)
            in_bounds_value = tf.ones_like(distances_to_origin) * 0.0
            distances_out_of_bounds = tf.where(out_of_bounds, distances_to_origin, in_bounds_value)
            out_of_bounds_loss = tf.reduce_mean(distances_out_of_bounds)
            return tf.norm(out_of_bounds_loss) * self.beta

        sdf_flat = Reshape(target_shape=[self.sdf_shape[0] * self.sdf_shape[1]])(sdf)
        sdf_gradient_flat = Reshape(target_shape=[self.sdf_shape[0] * self.sdf_shape[1] * 2])(sdf_gradient)

        fc_h = rope_input
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='tanh')(fc_h)
        self.sdf_input_layer = Dense(2, activation=None, activity_regularizer=oob_regularization)
        sdf_input = self.sdf_input_layer(fc_h)

        sdf_func_inputs = Concatenate()([sdf_flat, sdf_gradient_flat, sdf_resolution, sdf_origin, sdf_input])

        signed_distance = SDFLookup(self.sdf_shape)(sdf_func_inputs)
        negative_signed_distance = Lambda(lambda x: -x)(signed_distance)
        sigmoid_scale = args_dict['sigmoid_scale']
        bias = BiasLayer()(negative_signed_distance)
        logits = Lambda(lambda x: sigmoid_scale * x)(bias)
        # threshold = 0.0
        # logits = Lambda(lambda x: threshold - x)(signed_distance)
        predictions = Activation('sigmoid', name='combined_output')(logits)

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.sdf_input_model = Model(inputs=self.model_inputs, outputs=self.sdf_input_layer.output)

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'beta': self.beta,
            'label_type': [label_type.name for label_type in label_types],
            'commandline': self.args_dict['commandline'],
            'sigmoid_scale': self.args_dict['sigmoid_scale'],
            'hidden_layer_dims': self.fc_layer_sizes,
        }
        return metadata

    def violated(self, observations, sdf_data):
        m = observations.shape[0]
        rope_configuration = observations
        sdf = np.tile(np.expand_dims(sdf_data.sdf, axis=2), [m, 1, 1, 1])
        sdf_gradient = np.tile(sdf_data.gradient, [m, 1, 1, 1])
        sdf_origin = np.tile(sdf_data.origin, [m, 1])
        sdf_resolution = np.tile(sdf_data.resolution, [m, 1])
        sdf_extent = np.tile(sdf_data.extent, [m, 1])
        inputs_dict = {
            'rope_configuration': rope_configuration,
            'sdf': sdf,
            'sdf_gradient': sdf_gradient,
            'sdf_origin': sdf_origin,
            'sdf_resolution': sdf_resolution,
            'sdf_extent': sdf_extent
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)

        self.sdf_input_model.set_weights(self.keras_model.get_weights())
        predicted_point = self.sdf_input_model.predict(inputs_dict)

        return predicted_violated, predicted_point

    def __str__(self):
        return "sdf model"


class EvaluateResult:

    def __init__(self, rope_configuration, predicted_point, predicted_violated, true_violated):
        self.rope_configuration = rope_configuration
        self.predicted_point = predicted_point
        self.predicted_violated = predicted_violated
        self.true_violated = true_violated


def test_single_prediction(sdf_data, model, threshold, rope_configuration):
    rope_configuration = rope_configuration.reshape(-1, 6)
    predicted_violated, predicted_point = model.violated(rope_configuration, sdf_data)
    predicted_point = predicted_point.squeeze()
    rope_configuration = rope_configuration.squeeze()
    head_x = rope_configuration[4]
    head_y = rope_configuration[5]
    row_col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, sdf_data.resolution, sdf_data.origin)
    true_violated = sdf_data.sdf[row_col] < threshold

    result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, true_violated)
    return result


def test_predictions(model, environment):
    rope_configurations = environment.rope_data['rope_configurations']
    constraint_labels = environment.rope_data['constraints']

    predicted_violateds, predicted_points = model.violated(rope_configurations, environment.sdf_data)

    m = rope_configurations.shape[0]
    results = np.ndarray([m], dtype=EvaluateResult)
    for i in range(m):
        rope_configuration = rope_configurations[i]
        predicted_point = predicted_points[i]
        predicted_violated = predicted_violateds[i]
        constraint_label = constraint_labels[i]
        result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, constraint_label)
        results[i] = result
    return results
