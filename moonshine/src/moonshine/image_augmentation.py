import tensorflow as tf


# CuRL and other papers suggest that random crop is the most important data augmentation for learning state/dynamics
# https://www.tensorflow.org/tutorials/images/data_augmentation

def augment(image, image_h: int, image_w: int):
    # flatten multiple batch dimensions, if they exist
    original_shape = image.shape
    original_batch_dims = original_shape[:-3].as_list()
    actual_image_shape = original_shape[-3:].as_list()
    n_channels = original_shape[-1]
    one_batch_shape = [-1] + actual_image_shape
    one_batch_image = tf.reshape(image, one_batch_shape)
    new_batch_size = one_batch_image.shape[0]

    # zoom a bit first
    s = 1.4
    zoomed_shape = [int(s * image_h), int(s * image_w)]
    image_zoomed = tf.image.resize(one_batch_image, zoomed_shape, preserve_aspect_ratio=True)
    # Add 6 pixels of padding
    image_cp = tf.image.resize_with_crop_or_pad(image_zoomed, image_h + 6, image_w + 6)
    # Random crop back to the original size. I believe this crops every image in the batch the same way, which should be fine
    # since the batches are assembled randomly, and is good because then the cropping is consistent over time,
    # since time is currently shoved into batch dimension for these operations
    image_rc = tf.image.random_crop(image_cp, size=[new_batch_size, image_h, image_w, n_channels])
    image_rb = tf.image.random_brightness(image_rc, max_delta=0.5)  # Random brightness
    out_image = tf.clip_by_value(image_rb, 0, 1)

    out_image = tf.reshape(out_image, original_batch_dims + [image_h, image_w, n_channels])
    return out_image
