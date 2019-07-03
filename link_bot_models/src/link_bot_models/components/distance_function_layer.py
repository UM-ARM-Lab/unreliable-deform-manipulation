import keras.backend as K
from keras.layers import Lambda, Activation, Conv2D

from link_bot_models.components.distance_matrix_layer import DistanceMatrix


def distance_function_layer(sigmoid_scale, n_points, output_name):
    # Define the layers used
    p = "distance_function_"
    distance_matrix_layer = DistanceMatrix()
    weighted_distance_layer = Conv2D(1, (n_points, n_points), activation=None, use_bias=True, name=p + 'weighted_distance')
    scaled_sigmoid = Lambda(lambda x: sigmoid_scale * K.squeeze(K.squeeze(x, 1), 1), name=p + 'scale')
    sigmoid = Activation('sigmoid', name=output_name)

    def forward(rope_input):
        # define the forward pass
        distances = distance_matrix_layer(rope_input)
        weighted_distance = weighted_distance_layer(distances)
        logits = scaled_sigmoid(weighted_distance)
        prediction = sigmoid(logits)
        return prediction

    return distance_matrix_layer, forward
