import tensorflow as tf
import tempfile

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.01)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def balance_dataset(dataset):
    positive_examples = dataset.filter(lambda x: x >= 0)
    negative_examples = dataset.filter(lambda x: x < 0)

    tmpfile = tempfile.NamedTemporaryFile()
    print('caching to {}'.format(tmpfile.name))
    negative_examples = negative_examples.cache(tmpfile.name)
    negative_examples = negative_examples.repeat()

    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)
    return balanced_dataset


example_dataset = tf.data.Dataset.range(-10, 100, 1)

balanced_dataset = balance_dataset(example_dataset)
batched_dataset = balanced_dataset.batch(4)

batch = next(iter(batched_dataset))
print(batch.numpy())
