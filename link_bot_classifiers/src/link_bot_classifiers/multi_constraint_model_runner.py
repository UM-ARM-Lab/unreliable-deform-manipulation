#!/usr/bin/env python
from __future__ import print_function

import keras.backend as K
import numpy as np
from keras.layers import Input, Concatenate, Lambda
from keras.models import Model

from link_bot_classifiers.base_classifier_runner import BaseClassifierRunner
from moonshine.distance_function_layer import distance_function_layer
from moonshine.sdf_function_layer import sdf_function_layer
from link_bot_classifiers.label_types import LabelType


class MultiConstraintModelRunner(BaseClassifierRunner):
    def __init__(self, args_dict):
        super(MultiConstraintModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']
        self.N = args_dict['N']
        self.sigmoid_scale = args_dict['sigmoid_scale']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']
        self.beta = args_dict['beta']

        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf_input')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        # Distance Function
        n_points = int(self.N / 2)
        distance_matrix_layer, distance_function = distance_function_layer(self.sigmoid_scale, n_points,
                                                                           LabelType.Overstretching.name)
        overstretching_prediction = distance_function(rope_input)

        # SDF Function
        sdf_input_layer, sdf_function = sdf_function_layer(self.sdf_shape, self.fc_layer_sizes, self.beta, self.sigmoid_scale,
                                                           LabelType.SDF.name)
        self.sdf_function_prediction = sdf_function(sdf, sdf_gradient, sdf_resolution, sdf_origin, rope_input)

        # Combine
        concat_predictions = Concatenate(name='prediction_concat')([self.sdf_function_prediction, overstretching_prediction])
        prediction = Lambda(lambda x: K.sum(x, axis=1, keepdims=True) - K.prod(x, axis=1, keepdims=True),
                            name=LabelType.Combined.name)(concat_predictions)

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.model_outputs = [prediction, self.sdf_function_prediction, overstretching_prediction]
        self.keras_model = Model(inputs=self.model_inputs, outputs=self.model_outputs)
        self.sdf_input_model = Model(inputs=self.model_inputs, outputs=sdf_input_layer.output)

        losses = {
            LabelType.Combined.name: 'binary_crossentropy',
            LabelType.SDF.name: 'binary_crossentropy',
            LabelType.Overstretching.name: 'binary_crossentropy',
        }
        self.keras_model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])

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
