import numpy as np


class RigidTranslationModel:

    def __init__(self, beta, dt):
        self.beta = beta
        self.dt = dt

    def predict(self, states, actions):
        points = np.reshape(states, [-1, 3, 2])
        s_0 = points[0]
        s_t = s_0
        prediction = [s_0]
        for action in actions:
            # I've tuned beta on the no_obj_new training set based on the total error
            s_t = s_t + np.reshape(np.tile(np.eye(2), [3, 1]) @ action, [3, 2]) * self.dt * self.beta
            prediction.append(s_t)
        prediction = np.array(prediction)
        return prediction
