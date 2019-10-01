import tensorflow.keras.layers as layers


def simple_cnn_layer(conv_filters, use_bias=True):
    conv_layers = []
    for conv_filter in conv_filters:
        n_filters = conv_filter[0]
        filter_size = conv_filter[1]
        conv_layers.append((layers.Conv2D(n_filters, filter_size, activation='relu', use_bias=use_bias),
                            layers.MaxPool2D(2)))

    def forward(input_image):
        conv_h = input_image
        for conv, pool in conv_layers:
            conv_z = conv(conv_h)
            conv_h = pool(conv_z)
        return conv_h

    return forward


def simple_cnn_relu_layer(conv_filters, fc_layer_sizes, use_bias=True):
    conv_layers = []
    for conv_filter in conv_filters:
        n_filters = conv_filter[0]
        filter_size = conv_filter[1]
        conv_layers.append((layers.Conv2D(n_filters, filter_size, activation='relu', use_bias=use_bias),
                            layers.MaxPool2D(2)))

    flatten = layers.Flatten()

    dense_layers = []
    for fc_layer_size in fc_layer_sizes:
        dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))

    def forward(input_image):
        conv_h = input_image
        for conv, pool in conv_layers:
            conv_z = conv(conv_h)
            conv_h = pool(conv_z)

        conv_output = flatten(conv_h)

        fc_h = conv_output
        for dense in dense_layers:
            fc_h = dense(fc_h)
        output = fc_h
        return output

    return forward
