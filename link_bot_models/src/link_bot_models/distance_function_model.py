from __future__ import division, print_function, absolute_import

import numpy as np
import keras.backend as K
from keras import Model
from keras.layers import Input, Lambda, Conv2D, Activation

from link_bot_models.base_model import BaseModel
from link_bot_models.components.distance_function_layer import DistanceFunctionLayer
from link_bot_models.components.distance_matrix_layer import DistanceMatrix


class DistanceFunctionModel(BaseModel):

    def __init__(self, args_dict, N):
        super(DistanceFunctionModel, self).__init__(args_dict, N)

        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        layer = DistanceFunctionLayer(args_dict['sigmoid_scale'])
        prediction1 = layer(rope_input)
        prediction = Lambda(lambda x: x, name='combined_output')(prediction1)

        self.model_inputs = [rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=prediction)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        distance_matrix = self.keras_model.layers[-2].distance_matrix_layer.output
        self.distance_matrix_model = Model(inputs=self.model_inputs, outputs=distance_matrix)

    def metadata(self, label_types):
        metadata = {
            'sigmoid_scale': self.args_dict['sigmoid_scale'],
        }
        metadata.update(super(DistanceFunctionModel, self).metadata(label_types))
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
