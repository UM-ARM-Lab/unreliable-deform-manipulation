#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from keras.layers import Input, Concatenate, Dense
from keras.models import Model

from link_bot_classifiers.base_classifier_runner import BaseClassifierRunner
from moonshine.raster_points_layer import RasterPoints
from moonshine.simple_cnn_layer import simple_cnn_relu_layer


class RasterCNNModelRunner(BaseClassifierRunner):

    def __init__(self, args_dict, inputs, steps_per_epoch):
        super(RasterCNNModelRunner, self).__init__(args_dict, inputs, steps_per_epoch)
        self.sdf_shape = args_dict['sdf_shape']
        self.conv_filters = args_dict['conv_filters']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']
        sdf_shape = inputs['sdf'].shape[1:3].as_list()

        sdf = Input(tensor=inputs['sdf'])
        rope_config = Input(tensor=inputs['rope_configurations'])
        sdf_resolution = Input(tensor=inputs['sdf_resolution'])
        sdf_origin = Input(tensor=inputs['sdf_origin'])

        rope_image = RasterPoints(sdf_shape)([rope_config, sdf_resolution, sdf_origin])
        combined_image = Concatenate()([sdf, rope_image])

        cnn_output = simple_cnn_relu_layer(self.conv_filters, self.fc_layer_sizes)(combined_image)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(cnn_output)

        self.model_inputs = [sdf, rope_config, sdf_resolution, sdf_origin]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'],
                                 target_tensors=[inputs['constraints']])

    def violated(self):
        # data is none here because the model "has no inputs" in the sense that it's wired directly to the tf dataset
        predicted_violated = (self.keras_model.predict(x=None, steps=self.steps_per_epoch) > 0.5).astype(np.bool)
        return predicted_violated
