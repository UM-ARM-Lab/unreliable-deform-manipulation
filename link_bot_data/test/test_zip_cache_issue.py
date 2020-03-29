from time import perf_counter

import tensorflow as tf
import tempfile

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.01)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def make_cache_name():
    tmpfile = tempfile.NamedTemporaryFile()
    return tmpfile.name


def balance_dataset(dataset, cache):
    positive_examples = dataset.filter(lambda x: x >= 0)
    negative_examples = dataset.filter(lambda x: x < 0)

    if cache:
        cache_name = make_cache_name()
        print('caching to {}'.format(cache_name))
        negative_examples = negative_examples.cache(cache_name)
    else:
        print('no caching')
    negative_examples = negative_examples.repeat()

    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)
    return balanced_dataset


# no issues here
def test_no_cache():
    print("TEST NO CACHE")
    example_dataset = tf.data.Dataset.range(-10, 1000, 1)
    balanced_dataset = balance_dataset(example_dataset, cache=False)
    batched_dataset = balanced_dataset.batch(4)

    t0 = perf_counter()
    for batch in batched_dataset:
        pass
    dt = perf_counter() - t0
    print(dt)


# no issues, much faster
def test_cache():
    print("TEST CACHE")
    example_dataset = tf.data.Dataset.range(-10, 1000, 1)
    fast_balanced_dataset = balance_dataset(example_dataset, cache=True)
    fast_batched_dataset = fast_balanced_dataset.batch(4)

    t0 = perf_counter()
    for batch in fast_batched_dataset:
        pass
    dt = perf_counter() - t0
    print(dt)


# still no issues, but slow
def pre_cache_only():
    print("PRE-CACHE ONLY")
    example_dataset = tf.data.Dataset.range(-10, 1000, 1)
    cache_name = make_cache_name()
    print("pre-caching to {}".format(cache_name))
    example_dataset = example_dataset.cache(cache_name)
    fast_balanced_dataset = balance_dataset(example_dataset, cache=False)
    fast_batched_dataset = fast_balanced_dataset.batch(4)

    t0 = perf_counter()
    for batch in fast_batched_dataset:
        pass
    dt = perf_counter() - t0
    print(dt)


# THIS WILL CAUSE THE PROBLEM
def double_cache():
    print("CACHE TWICE, ERROR!")
    example_dataset = tf.data.Dataset.range(-10, 1000, 1)
    cache_name = make_cache_name()
    print("pre-caching to {}".format(cache_name))
    example_dataset = example_dataset.cache(cache_name)
    fast_balanced_dataset = balance_dataset(example_dataset, cache=True)
    fast_batched_dataset = fast_balanced_dataset.batch(4)

    t0 = perf_counter()
    for batch in fast_batched_dataset:
        pass
    dt = perf_counter() - t0
    print(dt)


test_no_cache()
test_cache()
pre_cache_only()
double_cache()
