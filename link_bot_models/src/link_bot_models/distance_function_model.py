from __future__ import division, print_function, absolute_import

import keras.backend as K
import numpy as np
import tensorflow as tf
import keras
from attr import dataclass
from keras.layers import Input, Conv2D, Lambda, Activation
from keras.models import Model

from link_bot_models.base_model import BaseModel
from link_bot_models.ops.distance_matrix_layer import DistanceMatrix
from link_bot_pycommon import link_bot_pycommon


class DistanceFunctionModel(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(DistanceFunctionModel, self).__init__(args_dict, sdf_shape, N)

        # we have to flatten everything in order to pass it around and I don't understand why
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        distances = DistanceMatrix()(rope_input)
        n_points = int(self.N / 2)
        self.l2reg = 1.0
        l2reg = self.l2reg
        conv = Conv2D(1, (n_points, n_points), activation=None, use_bias=True,
                      activity_regularizer=keras.regularizers.l2(l2reg))
        z = conv(distances)
        self.sigmoid_scale = 100
        sigmoid_scale = self.sigmoid_scale
        z = Lambda(lambda x: K.squeeze(x, 1), name='squeeze1')(z)
        logits = Lambda(lambda x: sigmoid_scale * K.squeeze(x, 1), name='squeeze2')(z)

        # TODO: this model doesn't handle "or" like conditions on the distances, since it's doing a linear combination
        predictions = Activation('sigmoid', name='combined_output')(logits)

        self.model_inputs = [rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        self.temp_model = Model(inputs=self.model_inputs, outputs=conv.output)

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'l2deg': self.l2reg,
            'sigmoid_scale': self.sigmoid_scale,
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

        self.temp_model.set_weights(self.keras_model.get_weights())
        predicted_point = self.temp_model.predict(inputs_dict)

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
