from keras.layers import Dense


def relu_layers(fc_layer_sizes, use_bias=True):
    dense_layers = []
    for fc_layer_size in fc_layer_sizes:
        dense_layers.append(Dense(fc_layer_size, activation='relu', use_bias=use_bias))

    def forward(x):
        fc_h = x
        for dense in dense_layers:
            fc_h = dense(fc_h)
        output = fc_h
        return output

    return forward
