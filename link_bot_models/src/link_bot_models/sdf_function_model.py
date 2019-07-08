import numpy as np
from keras.layers import Input
from keras.models import Model

from link_bot_models.base_model_runner import BaseModelRunner
from link_bot_models.components.sdf_function_layer import sdf_function_layer
from link_bot_models.label_types import LabelType


class SDFFunctionModelRunner(BaseModelRunner):
    def __init__(self, args_dict):
        super(SDFFunctionModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']

        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf_input')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        self.fc_layer_sizes = args_dict['fc_layer_sizes']
        self.beta = args_dict['beta']
        self.sigmoid_scale = args_dict['sigmoid_scale']

        sdf_input_layer, sdf_function = sdf_function_layer(self.sdf_shape, self.fc_layer_sizes, self.beta, self.sigmoid_scale,
                                                           LabelType.SDF.name)
        prediction = sdf_function(sdf, sdf_gradient, sdf_resolution, sdf_origin, rope_input)

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=prediction)
        self.sdf_input_model = Model(inputs=self.model_inputs, outputs=sdf_input_layer.output)

        losses = {
            LabelType.SDF.name: 'binary_crossentropy',
        }
        self.keras_model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])

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
            'sdf_input': sdf,
            'sdf_gradient': sdf_gradient,
            'sdf_origin': sdf_origin,
            'sdf_resolution': sdf_resolution,
            'sdf_extent': sdf_extent
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)
        self.sdf_input_model.set_weights(self.keras_model.get_weights())
        predicted_point = self.sdf_input_model.predict(inputs_dict)

        return predicted_violated, predicted_point


class EvaluateResult:

    def __init__(self, rope_configuration, predicted_point, predicted_violated, true_violated):
        self.rope_configuration = rope_configuration
        self.predicted_point = predicted_point
        self.predicted_violated = predicted_violated
        self.true_violated = true_violated

    def __str__(self):
        return "{} {} {} {} ".format(self.rope_configuration, self.predicted_point, self.predicted_violated, self.true_violated)


def test_single_prediction(sdf_data, model, rope_configuration):
    rope_configuration = rope_configuration.reshape(-1, 6)
    predicted_violated, predicted_point = model.violated(rope_configuration, sdf_data)
    predicted_point = predicted_point.squeeze()
    rope_configuration = rope_configuration.squeeze()

    result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, true_violated)
    return result


def test_predictions(model, environment):
    # TODO: redesign & rewrite all this visualization/plotting code
    rope_configurations = environment.rope_data['rope_configurations']
    constraint_labels = environment.rope_data[LabelType.Combined.name]

    predicted_violateds, predicted_points = model.violated(rope_configurations, environment.sdf_data)

    m = rope_configurations.shape[0]
    results = np.ndarray([m], dtype=EvaluateResult)
    for i in range(m):
        rope_configuration = rope_configurations[i]
        predicted_point = predicted_points[i]
        predicted_violated = predicted_violateds[i]
        constraint_label = constraint_labels[i]
        result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, constraint_label)
        results[i] = result
    return results
