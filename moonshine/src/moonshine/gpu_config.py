import tensorflow as tf


def limit_gpu_mem(gigs: float):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu = gpus[0]
    config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * gigs)]
    tf.config.experimental.set_virtual_device_configuration(gpu, config)
