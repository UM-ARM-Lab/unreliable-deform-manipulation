# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.ops import summary_op_util
from tensorflow.python.ops import summary_ops_v2

from moonshine import ffmpeg_gif


def py_gif_summary(tag, images, max_outputs, fps):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      tag: Name of the summary.
      images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
    Returns:
      The serialized `Summary` protocol buffer.
    Raises:
      ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
    """
    is_bytes = isinstance(tag, bytes)
    if is_bytes:
        tag = tag.decode("utf-8")
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise ValueError("Tensor must have dtype uint8 for gif summary.")
    if images.ndim != 5:
        raise ValueError("Tensor must be 5-D for gif summary.")
    batch_size, _, height, width, channels = images.shape
    if channels not in (1, 3):
        raise ValueError("Tensors must have 1 or 3 channels for gif summary.")

    summ = tf.Summary()
    num_outputs = min(batch_size, max_outputs)
    for i in range(num_outputs):
        image_summary = tf.Summary.Image()
        image_summary.height = height
        image_summary.width = width
        image_summary.colorspace = channels  # 1: grayscale, 3: RGB
        try:
            image_summary.encoded_image_string = ffmpeg_gif.encode_gif(images[i], fps)
        except (IOError, OSError) as e:
            tf.logging.warning(
                "Unable to encode images to a gif string because either ffmpeg is "
                "not installed or ffmpeg returned an error: %s. Falling back to an "
                "image summary of the first frame in the sequence.", e)
            try:
                from PIL import Image  # pylint: disable=g-import-not-at-top
                import io  # pylint: disable=g-import-not-at-top
                with io.BytesIO() as output:
                    Image.fromarray(images[i][0]).save(output, "PNG")
                    image_summary.encoded_image_string = output.getvalue()
            except:
                tf.logging.warning(
                    "Gif summaries requires ffmpeg or PIL to be installed: %s", e)
                image_summary.encoded_image_string = "".encode('utf-8') if is_bytes else ""
        if num_outputs == 1:
            summ_tag = "{}/gif".format(tag)
        else:
            summ_tag = "{}/gif/{}".format(tag, i)
        summ.value.add(tag=summ_tag, image=image_summary)
    summ_str = summ.SerializeToString()
    return summ_str


def gif_summary(name, tensor, max_outputs=3, fps=10, collections=None,
                family=None):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      name: Name of the summary.
      tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
      collections: Optional list of tf.GraphKeys.  The collections to add the
        summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tensor = tf.convert_to_tensor(tensor)
    if summary_op_util.skip_summary():
        return tf.constant("")
    with summary_op_util.summary_scope(
            str(name), family, values=[tensor]) as (tag, scope):
        tag = tf.convert_to_tensor(tag)
        max_outputs = tf.convert_to_tensor(max_outputs)
        fps = tf.convert_to_tensor(fps)
        val = tf.py_func(
            py_gif_summary,
            [tag, tensor, max_outputs, fps],
            tf.string,
            stateful=False,
            name=scope)
        summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
    return val


def gif_summary_v2(name, tensor, max_outputs, fps, family=None, step=None):
    def py_gif_event(step, tag, tensor, max_outputs, fps):
        summary = py_gif_summary(tag, tensor, max_outputs, fps)

        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ

        event = event_pb2.Event(summary=summary)
        event.wall_time = time.time()
        event.step = step
        event_pb = event.SerializeToString()
        return event_pb

    def function(tag, scope):
        # Note the identity to move the tensor to the CPU.
        tag_tensor = tf.convert_to_tensor(tag)
        max_outputs_tensor = tf.convert_to_tensor(max_outputs)
        fps_tensor = tf.convert_to_tensor(fps)
        input_tensors = [_choose_step(step), tag_tensor, tf.identity(tensor), max_outputs_tensor, fps_tensor]
        event = tf.py_func(
            py_gif_event,
            input_tensors,
            tf.string,
            stateful=False)
        return summary_ops_v2.import_event(event, name=scope)

    return summary_ops_v2.summary_writer_function(
        name, tensor, function, family=family)


def _choose_step(step):
    if step is None:
        return tf.train.get_or_create_global_step()
    if not isinstance(step, tf.Tensor):
        return tf.convert_to_tensor(step, tf.int64)
    return step
