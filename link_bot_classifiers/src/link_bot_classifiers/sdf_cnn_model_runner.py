#!/usr/bin/env python
from __future__ import print_function

import keras.backend as K
from keras.layers import Input, Concatenate, Lambda, Dense
from keras.models import Model

from link_bot_classifiers.base_classifier_runner import BaseClassifierRunner
from link_bot_classifiers.components.sdf_function_layer import sdf_function_layer
from link_bot_classifiers.components.simple_cnn_layer import simple_cnn_relu_layer
from link_bot_classifiers.label_types import LabelType


class SDFCNNModelRunner(BaseClassifierRunner):
    def __init__(self, args_dict):
        super(SDFCNNModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']
        self.N = args_dict['N']
        self.sigmoid_scale = args_dict['sigmoid_scale']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']
        self.conv_filters = args_dict['conv_filters']
        self.cnn_fc_layer_sizes = args_dict['cnn_fc_layer_sizes']
        self.beta = args_dict['beta']

        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf_input')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')
        rope_image = Input(shape=(self.sdf_shape[0], self.sdf_shape[1], 3), dtype='float32', name='rope_image')
        combined_image = Concatenate()([sdf, rope_image])

        # SDF Function
        sdf_input_layer, sdf_function = sdf_function_layer(self.sdf_shape, self.fc_layer_sizes, self.beta, self.sigmoid_scale,
                                                           LabelType.SDF.name)
        self.sdf_function_prediction = sdf_function(sdf, sdf_gradient, sdf_resolution, sdf_origin, rope_input)

        # CNN Bit
        cnn_output = simple_cnn_relu_layer(self.conv_filters, self.cnn_fc_layer_sizes)(combined_image)
        cnn_prediction = Dense(1, activation='sigmoid', name=LabelType.CNN_SDF.name)(cnn_output)

        # Combine
        concat_predictions = Concatenate(name='prediction_concat')([self.sdf_function_prediction, cnn_prediction])
        prediction = Lambda(lambda x: K.sum(x, axis=1, keepdims=True) - K.prod(x, axis=1, keepdims=True),
                            name=LabelType.Combined.name)(concat_predictions)

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input, rope_image]
        self.model_outputs = [prediction, self.sdf_function_prediction, cnn_prediction]
        self.keras_model = Model(inputs=self.model_inputs, outputs=self.model_outputs)
        self.sdf_input_model = Model(inputs=self.model_inputs, outputs=sdf_input_layer.output)

        losses = {
            LabelType.Combined.name: 'binary_crossentropy',
            LabelType.SDF.name: 'binary_crossentropy',
            LabelType.CNN_SDF.name: 'binary_crossentropy',
        }
        self.keras_model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])
