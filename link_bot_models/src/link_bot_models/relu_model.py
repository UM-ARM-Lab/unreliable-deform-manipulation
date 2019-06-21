from __future__ import division, print_function, absolute_import

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Activation, Lambda
from keras.models import Model

from link_bot_models.base_model import BaseModel


class ReluModel(BaseModel):

    def __init__(self, args_dict, N):
        super(ReluModel, self).__init__(args_dict, N)

        rope_input = Input(shape=(N,), dtype='float32', name='rope_configuration')

        self.fc_layer_sizes = [
            4096,
            4096,
            4096,
        ]

        fc_h = rope_input
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        logits = Dense(1, activation=None)(fc_h)
        sigmoid_scale = args_dict['sigmoid_scale']
        scaled_logits = Lambda(lambda x: sigmoid_scale * x)(logits)
        predictions = Activation('sigmoid', name='combined_output')(scaled_logits)

        # This creates a model that includes
        self.model_inputs = [rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'keras_version': str(keras.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'label_type': [label_type.name for label_type in label_types],
            'sigmoid_scale': self.args_dict['sigmoid_scale'],
            'fc_layer_sizes': self.fc_layer_sizes,
            'commandline': self.args_dict['commandline'],
        }
        return metadata

    def violated(self, observations):
        rope_configuration = observations
        inputs_dict = {
            'rope_configuration': rope_configuration,
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)

        return predicted_violated

    def __str__(self):
        return "keras relu model"
