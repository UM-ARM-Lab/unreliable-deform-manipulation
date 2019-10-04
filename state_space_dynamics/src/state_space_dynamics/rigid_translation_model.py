import numpy as np


class RigidTranslationModel:

    def __init__(self, beta, dt):
        self.beta = beta
        self.dt = dt
        self.n_state = 6
        self.n_control = 2

    def predict(self, first_states, batch_actions):
        """
        It's T+1 because it includes the first state
        :param np_first_states: [batch, 6]
        :param np_actions: [batch, T, 2]
        :return: [batch, T+1, 3, 2]
        """
        predictions = []
        for first_state, actions in zip(first_states, batch_actions):
            s_0 = np.reshape(first_state, [3, 2])
            prediction = [s_0]
            s_t = s_0
            for action in actions:
                # I've tuned beta on the no_obj_new training set based on the total error
                B = np.tile(np.eye(2), [3, 1])
                s_t = s_t + np.reshape(B @ action, [3, 2]) * self.dt * self.beta
                prediction.append(s_t)
            prediction = np.array(prediction)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions
