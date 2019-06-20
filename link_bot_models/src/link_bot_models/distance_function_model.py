from __future__ import division, print_function, absolute_import

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Activation
from keras.models import Model

from link_bot_models.base_model import BaseModel
from link_bot_models.ops.distance_matrix_layer import DistanceMatrix


class DistanceFunctionModel(BaseModel):

    def __init__(self, args_dict, N):
        super(DistanceFunctionModel, self).__init__(args_dict, N)

        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        distances = DistanceMatrix()(rope_input)
        n_points = int(self.N / 2)
        conv = Conv2D(1, (n_points, n_points), activation=None, use_bias=True)
        z = conv(distances)
        self.sigmoid_scale = 1
        sigmoid_scale = self.sigmoid_scale
        z = Lambda(lambda x: K.squeeze(x, 1), name='squeeze1')(z)
        logits = Lambda(lambda x: sigmoid_scale * K.squeeze(x, 1), name='squeeze2')(z)

        # TODO: this model doesn't handle "or" like conditions on the distances, since it's doing a linear combination
        predictions = Activation('sigmoid', name='combined_output')(logits)

        self.model_inputs = [rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        self.distance_matrix_model = Model(inputs=self.model_inputs, outputs=conv.input)

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'sigmoid_scale': self.sigmoid_scale,
            'label_type': [label_type.name for label_type in label_types],
            'commandline': self.args_dict['commandline'],
        }
        return metadata

    def violated(self, observations):
        rope_configuration = observations
        inputs_dict = {
            'rope_configuration': rope_configuration,
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)

        self.distance_matrix_model.set_weights(self.keras_model.get_weights())
        predicted_point = self.distance_matrix_model.predict(inputs_dict)

        return predicted_violated, predicted_point

    def __str__(self):
        return "distance function model"
