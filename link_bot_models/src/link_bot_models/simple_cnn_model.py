#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from keras.layers import Input, Concatenate, Dense
from keras.models import Model

from link_bot_models.base_model_runner import BaseModelRunner
from link_bot_models.components.action_smear_layer import action_smear_layer
from link_bot_models.components.simple_cnn_layer import simple_cnn_layer, simple_cnn_relu_layer


class SimpleCNNModelRunner(BaseModelRunner):

    def __init__(self, args_dict, inputs, steps_per_epoch):
        super(SimpleCNNModelRunner, self).__init__(args_dict, inputs, steps_per_epoch)
        self.sdf_shape = args_dict['sdf_shape']
        self.conv_filters_1 = args_dict['conv_filters_1']
        self.conv_filters_2 = args_dict['conv_filters_2']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']

        sdf = Input(tensor=inputs['sdf'])
        sdf_resolution = Input(tensor=inputs['sdf_resolution'])
        sdf_origin = Input(tensor=inputs['sdf_origin'])
        action = Input(tensor=inputs['actions'])

        action_image = action_smear_layer(action, sdf)(action)
        concat = Concatenate(axis=-1)([sdf, action_image])
        conv_h1 = simple_cnn_layer(self.conv_filters_1)(concat)
        out_h = simple_cnn_relu_layer(self.conv_filters_2, self.fc_layer_sizes)(conv_h1)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(out_h)

        self.model_inputs = [sdf, sdf_resolution, sdf_origin, action]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'],
                                 target_tensors=[inputs['constraints']])

    def violated(self):
        # data is none here because the model "has no inputs" in the sense that it's wired directly to the tf dataset
        predicted_violated = (self.keras_model.predict(x=None, steps=self.steps_per_epoch) > 0.5).astype(np.bool)
        return predicted_violated
