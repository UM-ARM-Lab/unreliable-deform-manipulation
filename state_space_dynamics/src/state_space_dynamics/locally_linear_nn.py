import json
import os

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from colorama import Fore
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from link_bot_classifiers.callbacks import DebugCallback, StopAtAccuracy
from link_bot_classifiers.components.relu_layers import relu_layers
from link_bot_pycommon import experiments_util


class LocallyLinearNN:

    def __init__(self, hparams):
        self.initial_epoch = 0
        self.hparams = hparams
        input_sequence_length = hparams['sequence_length'] - 1
        n_dim = hparams['n_points'] * 2
        m_dim = hparams['n_control']

        states = layers.Input(name='states', shape=(input_sequence_length, n_dim))
        actions = layers.Input(name='actions', shape=(input_sequence_length, 2))

        elements_in_A = hparams['n_points'] * (2 * 2)
        elements_in_B = n_dim * m_dim

        concat = layers.Concatenate()
        # state here is ? x 3 x 2
        nn = relu_layers(hparams['fc_layer_sizes'])
        num_elements_in_linear_model = elements_in_A + elements_in_B
        h_to_params = layers.Dense(num_elements_in_linear_model, activation=None, name='constraints')

        def AB_params(_states, _actions):
            # concat state and action here
            _states = tf.squeeze(_states, squeeze_dims=2)
            _state_actions = concat([_states, _actions])
            out_h = nn(_state_actions)
            params = h_to_params(out_h)
            return params

        def dynamics(_dynamics_inputs):
            _states, _actions = _dynamics_inputs

            s_0_flat = tf.reshape(_states[:, 0], [-1, n_dim, 1])

            # _gen_states should be ? x T x 3 x 2
            _gen_states = [s_0_flat]
            for t in range(input_sequence_length):
                s_t_flat = _gen_states[-1]
                action_t = actions[:, t]

                params_t = AB_params(s_t_flat, action_t)

                A_t_params, B_t_params = tf.split(params_t, [elements_in_A, elements_in_B], axis=1)
                A_t_per_point = tf.split(A_t_params, hparams['n_points'], axis=1)
                A_t_params = tf.Print(A_t_params, [tf.shape(A_t_params)])
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
            stacked = tf.squeeze(stacked, squeeze_dims=3)
            return stacked

        gen_states = layers.Lambda(lambda args: dynamics(args), name='output_states')([states, actions])

        self.model_inputs = [actions, states]
        self.keras_model = models.Model(inputs=self.model_inputs, outputs=gen_states)
        self.keras_model.compile(optimizer=tf.train.AdamOptimizer(),
                                 loss='mse',
                                 metrics=['accuracy'],
                                 # run_eagerly=True
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
        model = LocallyLinearNN(model_hparams)
        return model

    def save(self, checkpoint_directory):
        pass
