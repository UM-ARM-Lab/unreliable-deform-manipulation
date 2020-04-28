import tensorflow as tf


def limit_gpu_mem(gigs: int):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * gigs)])
