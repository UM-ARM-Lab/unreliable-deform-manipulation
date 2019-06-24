from __future__ import division, print_function, absolute_import

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

from link_bot_models.base_model import BaseModel


class SimpleCNNModel(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(SimpleCNNModel, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        sdf_input = Input(shape=(sdf_shape[0], sdf_shape[1], 1), dtype='float32', name='sdf')
        rope_input = Input(shape=(N,), dtype='float32', name='rope_configuration')

        self.conv_filters = [
            (64, (7, 7)),
            (32, (5, 5)),
            (16, (3, 3)),
            (16, (3, 3)),
        ]

        self.fc_layer_sizes = [
            256,
            32,
        ]

        conv_h = sdf_input
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_z = Conv2D(n_filters, filter_size, activation='relu')(conv_h)
            conv_h = MaxPool2D(2)(conv_z)
        conv_output = Flatten()(conv_h)

        concat = keras.layers.concatenate([conv_output, rope_input])
        fc_h = concat
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(fc_h)

        # This creates a model that includes
        self.model_inputs = [sdf_input, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self, label_types):
        extra_metadata = {
            'conv_filters': self.conv_filters,
            'fc_layer_sizes': self.fc_layer_sizes,
            'sdf_shape': self.sdf_shape,
        }
        return super().metadata(label_types).update(extra_metadata)

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
        return predicted_violated

    def __str__(self):
        return "keras constraint cnn"
