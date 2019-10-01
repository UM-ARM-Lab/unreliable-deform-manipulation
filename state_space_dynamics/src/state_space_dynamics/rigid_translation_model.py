import numpy as np


class RigidTranslationModel:

    def __init__(self, beta, dt):
        self.beta = beta
        self.dt = dt
        self.n_state = 6
        self.n_control = 2

    def predict(self, first_states, actions):
        """
        note that input_sequence_length = input_sequence_length - 1
        :param first_states: [batch, 6]
        :param actions: [batch, input_sequence_length, 2]
        :return: [batch, sequence_length, 3, 2]
        """
        s_0 = np.reshape(first_states, [-1, 3, 2])
        s_t = s_0
        predictions = [s_0]
        for action in actions:
            # I've tuned beta on the no_obj_new training set based on the total error
            s_t = s_t + np.reshape(np.tile(np.eye(2), [3, 1]) @ action, [3, 2]) * self.dt * self.beta
            predictions.append(s_t)
        predictions = np.transpose(np.array(predictions), [1, 0, 2, 3])
        return predictions
