from geometry_msgs.msg import Twist
from god_plugin.msg import WorldState
import numpy as np
import rospy

class BaseModel:

    def __init__(self, N, M, L, seed=0):
        """
        N: dimensionality of the full state
        M: dimensionality in the reduced state
        L: dimensionality in the actions
        """
        self.N = N
        self.M = M
        self.L = L

        np.random.seed(seed)
        self.set_world_state = rospy.ServiceProxy("/gazebo/set_world_state", WorldState)
        self.step_world_state = rospy.ServiceProxy("/gazebo/step_world_state", Twist)

    def reduce(self, s):
        # applies no model reduction, simply return s
        return s

    def predict(self, o, u, dt=None):
        # ros service call to my gazebo World Plugin,
        # which resets the world state to the state given in o (which is equal to s)
        # the applies the velocities given by u and returns the resulting s
        _ = self.set_world_state.call(o)
        next_o = self.step_world_state(u)
        return next_o

    def cost(self, o, g):
        return np.linalg.norm(o[0:2] - g[0:2])

    def save(self, outfile):
        pass

    def load(self, infile):
        pass

    def predict_from_s(self, s, u, dt=None):
        return self.predict(self.reduce(s), u, dt)

    def predict_from_o(self, o, u, dt=None):
        return self.predict(o, u, dt)

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def predict_cost_of_s(self, s, u, g):
        return self.cost(self.predict(self.reduce(s), u), g)

    def predict_cost(self, o, u, g):
        return self.cost(self.predict(o, u), g)

    def __repr__(self):
        pass
