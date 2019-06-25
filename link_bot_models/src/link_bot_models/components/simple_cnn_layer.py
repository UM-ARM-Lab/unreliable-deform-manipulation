from keras.layers import Conv2D, MaxPool2D, Flatten, Dense


def simple_cnn_layer(conv_filters, fc_layer_sizes):
    conv_layers = []
    for conv_filter in conv_filters:
        n_filters = conv_filter[0]
        filter_size = conv_filter[1]
        conv_layers.append((Conv2D(n_filters, filter_size, activation='relu'),
                            MaxPool2D(2)))

    flatten = Flatten()

    dense_layers = []
    for fc_layer_size in fc_layer_sizes:
        dense_layers.append(Dense(fc_layer_size, activation='relu'))

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
