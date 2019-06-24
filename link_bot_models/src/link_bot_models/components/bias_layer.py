from keras.layers import Layer


class BiasLayer(Layer):

    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        self.bias = None

    def build(self, input_shape):
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[1], 1),
                                    initializer='uniform',
                                    trainable=True)
        super(BiasLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        return x + self.bias

    def get_config(self):
        config = {
        }
        base_config = super(BiasLayer, self).get_config()
        return base_config.update(config)

    def compute_output_shape(self, input_shape):
        return input_shape
