from __future__ import division, print_function, absolute_import

import numpy as np
import keras.backend as K
import tensorflow as tf
from attr import dataclass
from keras.layers import Input, Conv2D, Lambda, Activation
from keras.models import Model
from keras.initializers import Constant

from link_bot_models.base_model import BaseModel
from link_bot_models.ops.distance_matrix_layer import DistanceMatrix
from link_bot_pycommon import link_bot_pycommon


class DistanceFunctionModel(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(DistanceFunctionModel, self).__init__(args_dict, sdf_shape, N)

        # we have to flatten everything in order to pass it around and I don't understand why
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        threshold = 0.5

        distances = DistanceMatrix()(rope_input)
        n_points = distances.shape[1]
        kernel_init_np = np.zeros((3, 3), dtype=np.float32)
        kernel_init_np[0, 1] = 1.0
        kernel_init_np[0, 2] = 1.0
        kernel_init = Constant(value=kernel_init_np)
        z = Conv2D(1, (n_points, n_points), activation=None, use_bias=False, kernel_initializer=kernel_init)(distances)
        z = Lambda(lambda x: K.squeeze(x, 1), name='squeeze1')(z)
        z = Lambda(lambda x: K.squeeze(x, 1), name='squeeze2')(z)

        logits = Lambda(lambda d: threshold - d, name='logits')(z)
        predictions = Activation('sigmoid', name='combined_output')(logits)

        self.model_inputs = [rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.beta = 1e-2

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'beta': self.beta,
            'label_type': [label_type.name for label_type in label_types],
            'commandline': self.args_dict['commandline'],
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


@dataclass
class EvaluateResult:
    rope_configuration: np.ndarray
    predicted_point: np.ndarray
    predicted_violated: bool
    true_violated: bool


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