from __future__ import division, print_function, absolute_import

import json
import os

from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import tensorflow as tf
from attr import dataclass
from colorama import Fore
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Lambda, Concatenate, Reshape, Activation
from keras.models import Model
from keras.models import load_model

from link_bot_models.tf_signed_distance_field_op import SDFLookup
from link_bot_pycommon import link_bot_pycommon, experiments_util


class ConstraintSDF:

    def __init__(self, args_dict, sdf_shape, N):
        self.args_dict = args_dict
        self.N = N

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))

        # we have to flatten everything in order to pass it around and I don't understand why
        sdf = Input(shape=[sdf_shape[0], sdf_shape[1], 1], dtype='float32', name='sdf')
        sdf_flat = Reshape(target_shape=[sdf_shape[0] * sdf_shape[1]])(sdf)
        sdf_gradient = Input(shape=[sdf_shape[0], sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_gradient_flat = Reshape(target_shape=[sdf_shape[0] * sdf_shape[1] * 2])(sdf_gradient)
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[N], dtype='float32', name='rope_configuration')

        self.fc_layer_sizes = [
            6,
        ]

        threshold_k = 0.0

        fc_h = rope_input
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu', use_bias=False)(fc_h)
        self.sdf_input_layer = Dense(2, activation=None, use_bias=True, name='sdf_input')
        sdf_input = self.sdf_input_layer(fc_h)

        sdf_func_inputs = Concatenate()([sdf_flat, sdf_gradient_flat, sdf_resolution, sdf_origin, sdf_input])

        signed_distance = SDFLookup(sdf_shape)(sdf_func_inputs)
        logits = Lambda(lambda d: threshold_k - d, name='logits')(signed_distance)
        predictions = Activation('sigmoid', name='combined_output')(logits)

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.sdf_input_model = Model(inputs=self.model_inputs,
                                         outputs=self.sdf_input_layer.output)
        self.sdf_input_model.compile(loss='mse', optimizer='adam')  # this is useless

        self.beta = 1e-2

        #     # FIXME: this assumes that the physical world coordinates (0,0) in meters is the origin/center of the SDF
        #     self.distances_to_origin = tf.norm(self.sdf_input, axis=1)
        #     oob_left = self.sdf_input[:, 0] <= self.sdf_extent[:, 0]
        #     oob_right = self.sdf_input[:, 0] >= self.sdf_extent[:, 1]
        #     oob_up = self.sdf_input[:, 1] <= self.sdf_extent[:, 2]
        #     oob_down = self.sdf_input[:, 1] >= self.sdf_extent[:, 3]
        #     self.out_of_bounds = tf.math.reduce_any(tf.stack((oob_up, oob_down, oob_left, oob_right), axis=1), axis=1, name='oob')
        #     self.in_bounds_value = tf.ones_like(self.distances_to_origin) * 0.0
        #     self.distances_out_of_bounds = tf.where(self.out_of_bounds, self.distances_to_origin, self.in_bounds_value)
        #     self.out_of_bounds_loss = tf.reduce_mean(self.distances_out_of_bounds, name='out_of_bounds_loss')

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'beta': self.beta,
            'label_type': [label_type.name for label_type in label_types],
            'commandline': self.args_dict['commandline'],
            'hidden_layer_dims': self.fc_layer_sizes,
        }
        return metadata

    def train(self, train_dataset, validation_dataset, label_types, epochs, log_path):
        callbacks = []
        if self.args_dict['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = self.metadata(label_types)
            metadata['log path'] = full_log_path
            metadata_file.write(json.dumps(metadata, indent=2))

            model_filename = os.path.join(full_log_path, "nn.{epoch:02d}.hdf5")

            checkpoint_callback = ModelCheckpoint(model_filename, monitor='loss', verbose=0, save_best_only=False,
                                                  save_weights_only=False, mode='auto', period=1)
            tensorboard = TensorBoard(log_dir=full_log_path)

            callbacks.append(checkpoint_callback)
            callbacks.append(tensorboard)

        train_generator = train_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
        validation_generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
        self.keras_model.fit_generator(train_generator, callbacks=callbacks, validation_data=validation_generator, epochs=epochs)
        self.evaluate(validation_dataset, label_types)

    def evaluate(self, validation_dataset, label_types, display=True):
        generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
        loss, accuracy = self.keras_model.evaluate_generator(generator)

        if display:
            print("Overall Loss: {:0.3f}".format(float(loss)))
        print("constraint prediction accuracy: {:4.1f}%".format(accuracy * 100))

        return loss, accuracy

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

        self.sdf_input_model.set_weights(self.keras_model.get_weights())
        predicted_point = self.sdf_input_model.predict(inputs_dict)

        return predicted_violated, predicted_point

    @staticmethod
    def load(args_dict):
        keras_model = load_model(args_dict['checkpoint'],
                                 custom_objects={'SDFLookup': SDFLookup})
        print(Fore.CYAN + "Restored keras model {}".format(args_dict['checkpoint']) + Fore.RESET)
        return keras_model

    def __str__(self):
        return "sdf model"


@dataclass
class EvaluateResult:
    rope_configuration: np.ndarray
    predicted_point: np.ndarray
    predicted_violated: bool
    true_violated: bool


def test_single_prediction(sdf_data, model, threshold, rope_configuration):
    rope_configuration = rope_configuration.reshape(-1, 6)
    predicted_violated, predicted_point = model.violated(rope_configuration, sdf_data)
    predicted_point = predicted_point.squeeze()
    rope_configuration = rope_configuration.squeeze()
    head_x = rope_configuration[4]
    head_y = rope_configuration[5]
    row_col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, sdf_data.resolution, sdf_data.origin)
    true_violated = sdf_data.sdf[row_col] < threshold

    result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, true_violated)
    return result


def test_predictions(model, environment):
    rope_configurations = environment.rope_data['rope_configurations']
    constraint_labels = environment.rope_data['constraints']

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
