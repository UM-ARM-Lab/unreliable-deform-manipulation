import os

import pydot
from keras.layers.wrappers import Wrapper
from keras.models import Sequential
from keras.utils import vis_utils


def model_to_dot(model,
                 show_shapes=False,
                 show_layer_names=True,
                 rankdir='TB'):
    """Convert a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.

    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """

    vis_utils._check_pydot()
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
    layers = model.layers

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                output_labels = str(layer.output_shape)
            except AttributeError:
                output_labels = 'multiple'
            if hasattr(layer, 'input_shape'):
                input_labels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                input_labels = ', '.join([str(ishape) for ishape in layer.input_shapes])
            else:
                input_labels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, input_labels, output_labels)
        color = '#aaffaa' if layer.trainable else '#ffaaaa'

        node = pydot.Node(layer_id, label=label, fillcolor=color, style='filled')
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot_model(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB'):
    """Converts a Keras model to dot format and save to a file.

    # Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    """
    dot = model_to_dot(model, show_shapes, show_layer_names, rankdir)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
