from __future__ import division, print_function, absolute_import

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Concatenate
from keras.models import Model

from link_bot_models.base_model import BaseModel


class ConstraintRasterCNN(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(ConstraintRasterCNN, self).__init__(args_dict, sdf_shape, N)

        sdf = Input(shape=(sdf_shape[0], sdf_shape[1], 1), dtype='float32', name='sdf')
        rope_image = Input(shape=(sdf_shape[0], sdf_shape[1], 3), dtype='float32', name='rope_image')
        combined_image = Concatenate()([sdf, rope_image])

        self.conv_filters = [
            (16, (3, 3)),
            (16, (3, 3)),
        ]

        self.fc_layer_sizes = [
            16,
            16,
        ]

        conv_h = combined_image
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_z = Conv2D(n_filters, filter_size)(conv_h)
            conv_h = MaxPool2D(2)(conv_z)

        conv_output = Flatten()(conv_h)

        fc_h = conv_output
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(fc_h)

        self.model_inputs = [sdf, rope_image]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'keras_version': str(keras.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'label_type': [label_type.name for label_type in label_types],
            'N': self.N,
            'conv_filters': self.conv_filters,
            'fc_layer_sizes': self.fc_layer_sizes,
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
        return predicted_violated

    def __str__(self):
        return "keras constraint raster cnn"
