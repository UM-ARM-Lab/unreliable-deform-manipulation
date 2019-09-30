import json
import os

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from colorama import Fore
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from link_bot_classifiers.callbacks import DebugCallback, StopAtAccuracy
from link_bot_classifiers.components.action_smear_layer import action_smear_layer
from link_bot_classifiers.components.raster_points_layer import RasterPoints
from link_bot_classifiers.components.simple_cnn_layer import simple_cnn_relu_layer
from link_bot_pycommon import experiments_util


class LocallyLinearCNN:

    def __init__(self, hparams):
        self.initial_epoch = 0
        self.hparams = hparams
        input_sequence_length = hparams['sequence_length'] - 1
        n_dim = hparams['n_points'] * 2

        states = layers.Input(name='states', shape=(input_sequence_length, n_dim))
        actions = layers.Input(name='actions', shape=(input_sequence_length, 2))
        # These are also used to define the image itself, not just the SDF
        sdf_resolution = layers.Input(name='sdf_resolution', shape=(input_sequence_length, 2))
        sdf_origin = layers.Input(name='sdf_origin', shape=(input_sequence_length, 2))

        if hparams['use_sdf']:
            sdf = layers.Input(name='sdf', shape=(input_sequence_length, hparams['sdf_shape'][0], hparams['sdf_shape'][1], 1))

            binary_sdf = layers.Lambda(function=lambda sdf: tf.cast(sdf > 0, dtype=tf.float32), name='make_binary')(sdf)
            action_image = action_smear_layer(actions, hparams['sdf_shape'][0], hparams['sdf_shape'][1])(actions)
        else:
            # NOTE: this we do ahead of time because it never changes
            action_image = action_smear_layer(actions, hparams['sdf_shape'][0], hparams['sdf_shape'][1])(actions)
        # TODO: How do we rasterize the rope configuration? should it have anything to do with the SDF?
        # it just needs to be the same dimensions...? but if there is no sdf shape then it can be arbitrarily chosen
        # so just choose a fixed size for the rope image if use_sdf is false?

        elements_in_A = hparams['n_points'] * (2 * 2)
        elements_in_B = n_dim * hparams['n_control']

        # state here is ? x 3 x 2
        cnn = simple_cnn_relu_layer(hparams['conv_filters'], hparams['fc_layer_sizes'])
        # right now we output N^2 + N*M elements for full A and B matrices
        num_elements_in_linear_model = elements_in_A + elements_in_B
        h_to_params = layers.Dense(num_elements_in_linear_model, activation='sigmoid', name='constraints')

        def AB_params(states):
            out_h = cnn(states)
            params = h_to_params(out_h)
            return params

        def dynamics(_dynamics_inputs):
            _states, _actions = _dynamics_inputs

            s_0_flat = tf.reshape(_states[:, 0], [-1, n_dim, 1])

            # _gen_states should be ? x T x 3 x 2
            _gen_states = [s_0_flat]
            for t in range(input_sequence_length):
                s_t_flat = _gen_states[-1]
                action_image_t = action_image[:, t]

                # First raster the previously predicted rope state into an image, then concat with smeared actions
                # then pass to the AB network
                # TODO: support concatenation with an SDF?
                rope_images = RasterPoints(hparams['sdf_shape'])([s_t_flat, sdf_resolution[:, t], sdf_origin[:, t]])
                print(action_image_t, rope_images)
                concat = layers.Concatenate(axis=-1)([action_image_t, rope_images])

                params_t = AB_params(concat)

                A_t_params, B_t_params = tf.split(params_t, [elements_in_A, elements_in_B], axis=1)
                A_t_per_point = tf.split(A_t_params, hparams['n_points'], axis=1)
                B_t_per_point = tf.split(B_t_params, hparams['n_points'], axis=1)
                A_t_per_point = [tf.linalg.LinearOperatorFullMatrix(tf.reshape(_a_p, [-1, 2, 2])) for _a_p
                                 in A_t_per_point]
                B_t_per_point = [tf.reshape(_b_p, [-1, 2, 2]) for _b_p in B_t_per_point]
                A_t = tf.linalg.LinearOperatorBlockDiag(A_t_per_point).to_dense("A_t")
                B_t = tf.concat(B_t_per_point, axis=1)

                u_t = tf.expand_dims(_actions[:, t], axis=-1)

                s_t_plus_1_flat = tf.linalg.matmul(A_t, s_t_flat) + tf.linalg.matmul(B_t, u_t)

                _gen_states.append(s_t_plus_1_flat)

            stacked = tf.stack(_gen_states)
            stacked = tf.transpose(stacked, [1, 0, 2, 3])
            return stacked

        gen_states = layers.Lambda(lambda args: dynamics(args), name='output_states')([states, actions])

        if hparams['use_sdf']:
            self.model_inputs = [sdf, sdf_resolution, sdf_origin, actions, states]
        else:
            self.model_inputs = [sdf_resolution, sdf_origin, actions, states]
        self.keras_model = models.Model(inputs=self.model_inputs, outputs=gen_states)
        self.keras_model.compile(optimizer=tf.train.AdamOptimizer(),
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'],
                                 run_eagerly=True
                                 )

    def train(self, train_dataset, train_tf_dataset, val_dataset, val_tf_dataset, log_path, args):
        callbacks = []
        if args.log is not None:
            full_log_path = os.path.join("log_data", log_path)

            print(Fore.CYAN + "Logging to {}".format(full_log_path) + Fore.RESET)

            experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "hparams.json")
            with open(metadata_path, 'w') as metadata_file:
                metadata = self.hparams
                metadata['log path'] = full_log_path
                metadata_file.write(json.dumps(metadata, indent=2))

            model_filename = os.path.join(full_log_path, "nn.{epoch:02d}.hdf5")

            checkpoint_callback = ModelCheckpoint(model_filename, monitor='loss', save_weights_only=True)
            callbacks.append(checkpoint_callback)

            tensorboard = TensorBoard(log_dir=full_log_path)
            callbacks.append(tensorboard)

            val_acc_threshold = args.val_acc_threshold
            if val_acc_threshold is not None:
                if args.validation:
                    raise ValueError("Validation dataset must be provided in order to use this monitor")
                if val_acc_threshold < 0 or val_acc_threshold > 1:
                    raise ValueError("val_acc_threshold {} must be between 0 and 1 inclusive".format(val_acc_threshold))
                stop_at_accuracy = StopAtAccuracy(val_acc_threshold)
                callbacks.append(stop_at_accuracy)

            if args.early_stopping:
                if args.validation:
                    raise ValueError("Validation dataset must be provided in order to use this monitor")
                early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.001, verbose=args.verbose)
                callbacks.append(early_stopping)

            if args.debug:
                callbacks.append(DebugCallback())

        steps_per_epoch = train_dataset.num_examples_per_epoch() // args.batch_size
        val_steps_per_epoch = val_dataset.num_examples_per_epoch() // args.batch_size

        if not args.validation:
            val_tf_dataset = None
            val_steps_per_epoch = None

        self.keras_model.fit(x=train_tf_dataset,
                             y=None,
                             callbacks=callbacks,
                             initial_epoch=self.initial_epoch,
                             steps_per_epoch=steps_per_epoch,
                             validation_data=val_tf_dataset,
                             validation_steps=val_steps_per_epoch,
                             epochs=args.epochs,
                             verbose=True)

    def get_default_hparams(self):
        return {
        }

    @staticmethod
    def load(checkpoint_directory):
        hparams_path = checkpoint_directory / 'hparams.json'
        model_hparams = json.load(open(hparams_path, 'r'))
        model = LocallyLinearCNN(model_hparams)
        return model

    def save(self, checkpoint_directory):
        pass
