import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, Concatenate, Reshape

from link_bot_models.base_model import BaseModel

from link_bot_models.sdf_function_model import SDFFunctionLayer
from link_bot_models.distance_function_model import DistanceFunctionLayer


class MultiConstraintModel(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(MultiConstraintModel, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        # we have to flatten everything in order to pass it around and I don't understand why
        sdf_flat = Reshape(target_shape=[self.sdf_shape[0] * self.sdf_shape[1]])(sdf)
        sdf_gradient_flat = Reshape(target_shape=[self.sdf_shape[0] * self.sdf_shape[1] * 2])(sdf_gradient)

        self.sdf_fc_layer_sizes = [
            16,
            16,
        ]
        self.sdf_oob_beta = 1e-2
        sigmoid_scale = args_dict['sigmoid_scale']
        sdf_layer = SDFFunctionLayer(self.sdf_shape, self.sdf_fc_layer_sizes, self.sdf_oob_beta, sigmoid_scale)
        sdf_prediction = sdf_layer([sdf_flat, sdf_gradient_flat, sdf_resolution, sdf_origin, rope_input])
        distance_layer = DistanceFunctionLayer(sigmoid_scale)
        distance_model = distance_layer(rope_input)

        dist_prediction = distance_model(rope_input)
        predictions = Concatenate()([sdf_prediction, dist_prediction])

        combined_prediction = Activation(activation='softmax')(predictions)
        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=combined_prediction)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.individual_predictions_model = Model(inputs=self.model_inputs, outputs=predictions)

    def metadata(self, label_types):
        extra_metadata = {
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

        combined_prediction = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)

        self.individual_predictions_model.set_weights(self.keras_model.get_weights())
        individual_predictions = self.individual_predictions_model.predict(inputs_dict)

        return combined_prediction, individual_predictions

    def __str__(self):
        return "multi-constraint model"
