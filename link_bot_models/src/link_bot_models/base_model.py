import json
import os
from time import time

import numpy as np
import tensorflow as tf
from colorama import Fore
from tensorflow.python import debug as tf_debug

import link_bot_pycommon.experiments_util
from link_bot_models.exceptions import FinishSetupNotCalledInConstructor


class BaseModel:

    def __init__(self, args_dict, N):
        """
        args_dict: the argsparse args but as a dict
        N: dimensionality of the full state
        """
        self.args_dict = args_dict
        self.N = N

        # A bunch of variables we assume will be defined by subclasses
        self.sess = None
        self.saver = None
        self.global_step = None
        self.loss = None
        self.opt = None
        self.train_summary = None
        self.validation_summary = None

        self.finish_setup_called = False

        # add some default arguments
        # FIXME: add these to the command line parsers
        if 'gpu_memory_fraction' not in self.args_dict:
            self.args_dict['gpu_memory_fraction'] = 0.2

        self.seed = self.args_dict['seed']
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

    def finish_setup(self):
        self.train_summary = tf.summary.merge_all('train')
        self.validation_summary = tf.summary.merge_all('validation')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args_dict['gpu_memory_fraction'])
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        if self.args_dict['debug']:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.saver = tf.train.Saver(max_to_keep=None)

        self.finish_setup_called = True

    def setup(self):
        if not self.finish_setup_called:
            raise FinishSetupNotCalledInConstructor(type(self).__name__)

        if self.args_dict['checkpoint']:
            self.sess.run([tf.local_variables_initializer()])
            self.load()
        else:
            self.init()

    def init(self):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def train(self, train_dataset, validation_dataset, label_types, epochs, log_path, **kwargs):
        if not self.finish_setup_called:
            raise FinishSetupNotCalledInConstructor(type(self).__name__)

        interrupted = False

        writer = None
        full_log_path = None
        if self.args_dict['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            link_bot_pycommon.experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = self.metadata()
            metadata['log path'] = full_log_path
            metadata_file.write(json.dumps(metadata, indent=2))

            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            train_ops = [self.global_step, self.train_summary, self.loss, self.opt]
            validation_ops = [self.validation_summary, self.loss]

            self.start_train_hook()

            if self.args_dict['log'] is not None:
                self.save(full_log_path, self.args_dict['log'])

            train_generator = train_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
            validation_generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])

            step = self.sess.run(self.global_step)
            for epoch in range(epochs):
                ############
                # Training #
                ############

                self.train_init_epoch_hook(epoch, step)

                for train_batch_x, train_batch_y in train_generator:
                    train_feed_dict = self.build_feed_dict(train_batch_x, train_batch_y, **kwargs)

                    self.train_feed_hook(step, train_batch_x, train_batch_y)

                    step, train_summary, train_batch_loss, _ = self.sess.run(train_ops, feed_dict=train_feed_dict)

                    if step % self.args_dict['log_period'] == 0 or step == 1:
                        if self.args_dict['log'] is not None:
                            writer.add_summary(train_summary, step)

                    if step % self.args_dict['print_period'] == 0 or step == 1:
                        print('epoch {:4d}, step: {:4d}, batch loss: {:8.4f}'.format(epoch, step, train_batch_loss))

                ##############
                # Validation #
                ##############

                if epoch % self.args_dict['val_period'] == 0 or epoch == 0:

                    self.validation_init_hook(epoch, step)

                    val_loss_sum = 0
                    for validation_batch_x, validation_batch_y in validation_generator:
                        validation_feed_dict = self.build_feed_dict(validation_batch_x, validation_batch_y, **kwargs)
                        validation_summary, validation_loss = self.sess.run(validation_ops, feed_dict=validation_feed_dict)
                        val_loss_sum += validation_loss

                        if self.args_dict['log'] is not None:
                            writer.add_summary(validation_summary, step)

                    val_loss_mean = val_loss_sum / len(validation_generator)

                    validation_log_msg = 'epoch {:4d}, step: {:4d} loss: {:8.4f}'.format(epoch, step, val_loss_mean)
                    print(Fore.YELLOW + validation_log_msg + Fore.RESET)

                ##########
                # Saving #
                ##########

                if epoch % self.args_dict['save_period'] == 0 and epoch > 0:
                    if self.args_dict['log'] is not None:
                        self.save(full_log_path)

        except KeyboardInterrupt:
            print("stop!!!")
            interrupted = True
            pass
        finally:
            if self.args_dict['log'] is not None:
                self.save(full_log_path)

        return interrupted

    def build_feed_dict(self, x, y, **kwargs):
        raise NotImplementedError()

    def start_train_hook(self):
        pass

    def train_feed_hook(self, iteration, train_x_batch, train_y_batch):
        pass

    def load(self):
        self.saver.restore(self.sess, self.args_dict['checkpoint'])
        global_step = self.sess.run(self.global_step)
        print(
            Fore.CYAN + "Restored ckpt {} at step {:d}".format(self.args_dict['checkpoint'], global_step) + Fore.RESET)

    def save(self, log_path, log=True, loss=None):
        global_step = self.sess.run(self.global_step)
        if log:
            if loss is not None:
                print(Fore.CYAN + "Saving ckpt {} at step {:d} with loss {}".format(log_path, global_step,
                                                                                    loss) + Fore.RESET)
            else:
                print(Fore.CYAN + "Saving ckpt {} at step {:d}".format(log_path, global_step) + Fore.RESET)
        self.saver.save(self.sess, os.path.join(log_path, "nn.ckpt"), global_step=self.global_step)

    def metadata(self):
        raise NotImplementedError()

    def __str__(self):
        return "base_model"

    def train_init_epoch_hook(self, epoch, step):
        pass

    def validation_init_hook(self, epoch, step):
        pass
