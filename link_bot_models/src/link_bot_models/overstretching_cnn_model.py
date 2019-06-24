from __future__ import division, print_function, absolute_import

import keras
import numpy as np
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

from link_bot_models.base_model import BaseModel
from link_bot_pycommon import link_bot_pycommon


class OverstretchingCNNModel(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(OverstretchingCNNModel, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        rope_image = Input(shape=(sdf_shape[0], sdf_shape[1], 3), dtype='float32', name='rope_image')

        self.conv_filters = [
            (32, (7, 7)),
            (16, (5, 5)),
        ]

        self.fc_layer_sizes = [
            32,
            32,
        ]

        conv_h = rope_image
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_z = Conv2D(n_filters, filter_size, activation='relu')(conv_h)
            conv_h = MaxPool2D(2)(conv_z)

        conv_output = Flatten()(conv_h)

        fc_h = conv_output
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(fc_h)

        self.model_inputs = [rope_image]
        keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # self.keras_model = multi_gpu_model(keras_model, gpus=args_dict['n_gpus'])
        self.keras_model = keras_model

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
        rope_configurations = observations
        rope_images = link_bot_pycommon.make_rope_images(sdf_data, rope_configurations)
        inputs_dict = {
            'rope_image': rope_images,
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)
        return predicted_violated

    def __str__(self):
        return "keras constraint raster cnn"
