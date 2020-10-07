import tensorflow as tf


# CuRL and other papers suggest that random crop is the most important data augmentation for learning state/dynamics
# https://www.tensorflow.org/tutorials/images/data_augmentation

def resize_and_rescale(image, label, image_h, image_w):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [image_h, image_w])
    image = (image / 255.0)
    return image, label


def augment(image, label, image_h, image_w):
    image, label = resize_and_rescale(image, label, image_h, image_w)
    # Add 6 pixels of padding
    image = tf.image.resize_with_crop_or_pad(image, image_h + 6, image_w + 6)
    # Random crop back to the original size
    image = tf.image.random_crop(image, size=[image_h, image_w, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
    image = tf.clip_by_value(image, 0, 1)
    return image, label
