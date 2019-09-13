#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.models import Model

from link_bot_models.base_model_runner import BaseModelRunner
from link_bot_models.components.action_smear_layer import action_smear_layer
from link_bot_models.components.raster_points_layer import RasterPoints
from link_bot_models.components.simple_cnn_layer import simple_cnn_layer, simple_cnn_relu_layer


class SimpleCNNModelRunner(BaseModelRunner):

    def __init__(self, args_dict):
        super(SimpleCNNModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']
        self.conv_filters = args_dict['conv_filters']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']

        sdf = Input(name='sdf', shape=(self.sdf_shape[0], self.sdf_shape[1], 1))
        rope_config = Input(name='rope_configurations', shape=(self.N,))
        sdf_resolution = Input(name='sdf_resolution', shape=(2,))
        sdf_origin = Input(name='sdf_origin', shape=(2,))
        action = Input(name='actions', shape=(2,))

        action_image = action_smear_layer(action, sdf)(action)
        rope_image = RasterPoints(self.sdf_shape)([rope_config, sdf_resolution, sdf_origin])
        concat = Concatenate(axis=-1)([sdf, action_image, rope_image])
        out_h = simple_cnn_relu_layer(self.conv_filters, self.fc_layer_sizes)(concat)
        predictions = Dense(1, activation='sigmoid', name='constraints')(out_h)

        self.model_inputs = [sdf, sdf_resolution, sdf_origin, action, rope_config]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])

    def violated(self, steps_per_epoch):
        # data is none here because the model "has no inputs" in the sense that it's wired directly to the tf dataset
        predicted_violated = (self.keras_model.predict(x=None, steps=steps_per_epoch) > 0.5).astype(np.bool)
        return predicted_violated
