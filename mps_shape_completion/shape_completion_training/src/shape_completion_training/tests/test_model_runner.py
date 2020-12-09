import unittest

import tensorflow as tf

from shape_completion_training.model.utils import reduce_mean_dict
from shape_completion_training.model_runner import ModelRunner
from shape_completion_training.my_keras_model import MyKerasModel

params = {
    'learning_rate': 0.1,
}


class FakeModel(MyKerasModel):
    def __init__(self, params, batch_size=16):
        super().__init__(hparams=params, batch_size=batch_size)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, dataset_element, training=False, **kwargs):
        inputs, _ = dataset_element
        x = inputs['x']
        y = self.dense(x)
        return {'y': y}

    @tf.function
    def compute_loss(self, dataset_element, outputs):
        loss = tf.keras.losses.mse(dataset_element[1]['y'], outputs['y'])
        losses = {
            "loss": loss,
            "mock_loss": tf.constant(3),
        }
        return reduce_mean_dict(losses)


class ModelRunnerTraining(unittest.TestCase):
    @staticmethod
    def simple_dataset(a, b):
        numbers = tf.data.Dataset.range(a, b).map(lambda x: tf.cast(x, tf.float32))
        x = numbers.map(lambda x: {'x': x}).batch(1).batch(64)
        y = numbers.map(lambda x: {'y': x * 2 + 2}).batch(1).batch(64)
        dataset = tf.data.Dataset.zip((x, y))
        return dataset

    @classmethod
    def setUpClass(cls):
        cls.dataset = ModelRunnerTraining.simple_dataset(0, 100)
        cls.val_dataset = ModelRunnerTraining.simple_dataset(200, 300)

    def test_train(self):
        model = FakeModel(params=params)
        mr = ModelRunner(model=model, training=True, params=params, write_summary=False)
        mr.train(ModelRunnerTraining.dataset, ModelRunnerTraining.val_dataset, num_epochs=1)

    def test_save_latest_and_best(self):
        model = FakeModel(params=params)
        mr = ModelRunner(model=model, group_name='latest_and_best', training=True, params=params, write_summary=False)
        mr.train(ModelRunnerTraining.dataset, ModelRunnerTraining.val_dataset, num_epochs=100)

        latest_checkpoint_dir = mr.trial_path / 'latest_checkpoint' / 'checkpoint'
        best_checkpoint_dir = mr.trial_path / 'best_checkpoint' / 'checkpoint'
        self.assertTrue(latest_checkpoint_dir.exists())
        self.assertTrue(best_checkpoint_dir.exists())

        latest_checkpoint = mr.latest_checkpoint_manager.latest_checkpoint
        latest_ckpt = tf.train.Checkpoint(step=tf.Variable(1))
        latest_ckpt.restore(latest_checkpoint).expect_partial()
        latest_checkpoint_step = latest_ckpt.step.numpy()

        best_checkpoint = mr.best_checkpoint_manager.latest_checkpoint
        best_ckpt = tf.train.Checkpoint(step=tf.Variable(1))
        best_ckpt.restore(best_checkpoint).expect_partial()
        best_checkpoint_step = best_ckpt.step.numpy()
        self.assertLess(best_checkpoint_step, latest_checkpoint_step)


if __name__ == '__main__':
    unittest.main()
