from keras import Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense


class SimpleCNNModel(Model):

    def __init__(self, conv_filters, fc_layer_sizes, output_dim, **kwargs):
        super(SimpleCNNModel, self).__init__(**kwargs)
        self.conv_filters = conv_filters
        self.fc_layer_sizes = fc_layer_sizes
        self.output_dim = output_dim

    def call(self, input_image, **kwargs):
        conv_h = input_image
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_z = Conv2D(n_filters, filter_size, activation='relu')(conv_h)
            conv_h = MaxPool2D(2)(conv_z)

        conv_output = Flatten()(conv_h)

        fc_h = conv_output
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        outputs = Dense(self.output, activation='sigmoid')(fc_h)
        return outputs

    def get_config(self):
        config = {
            'conv_filters': self.conv_filters,
            'fc_layer_sizes': self.fc_layer_sizes,
            'output_dim': self.output_dim
        }
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
