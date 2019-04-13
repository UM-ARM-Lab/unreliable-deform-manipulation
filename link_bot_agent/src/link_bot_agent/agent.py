#!/usr/bin/env python
import numpy as np
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
import rospy


def h(v1, v2):
    return np.linalg.norm(np.array(v1.o) - np.array(v2.o))


class LinkConfig:

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None


def get_rope_data(get_link_state, num_links, control_link_i, fx, fy):
    # return value should be 3D array
    # first dim is position/velocity/force
    # second dim is link
    # third dim is x/y

    data = np.ndarray((3, num_links, 2))
    for link_idx in range(num_links):
        req = GetLinkStateRequest()
        req.link_name = "rope::link_{:d}".format(link_idx)
        response = get_link_state.call(req)
        data[0, link_idx, 0] = response.link_state.pose.position.x
        data[0, link_idx, 1] = response.link_state.pose.position.y
        data[1, link_idx, 0] = response.link_state.twist.linear.x
        data[1, link_idx, 1] = response.link_state.twist.linear.y
        if link_idx == control_link_i:
            data[2, link_idx, 0] = fx
            data[2, link_idx, 1] = fy
        else:
            data[2, link_idx, 0] = 0
            data[2, link_idx, 1] = 0

    return data


def get_state(get_link_state):
    links = {'link_0': LinkConfig(),
             'link_1': LinkConfig(),
             'head': LinkConfig()}

    # get the new states
    for link_name, link_config in links.items():
        req = GetLinkStateRequest()
        req.link_name = "link_bot::" + link_name
        response = get_link_state.call(req)
        link_config.x = response.link_state.pose.position.x
        link_config.y = response.link_state.pose.position.y
        link_config.vx = response.link_state.twist.linear.x
        link_config.vy = response.link_state.twist.linear.y

    return [
        links['link_0'].x,
        links['link_0'].y,
        links['link_1'].x,
        links['link_1'].y,
        links['head'].x,
        links['head'].y,
    ]


class GazeboAgent:

    def __init__(self, M, N, dt, model, gazebo_model_name):
        self.model_name = gazebo_model_name
        self.N = N
        self.M = M
        self.dt = dt
        self.model = model

        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

    def sample_action(self, o, goal):
        potential_actions = 0.3 * np.random.randn(250, 2, 1)
        min_cost_action = None
        next_o = None
        min_cost = 1e9
        xs = []
        ys = []
        for a in potential_actions:
            o_ = self.model.predict_from_o(o, a, dt=self.dt)
            c = self.model.cost(o_, goal)[0, 0]
            x = o_[0, 0]
            y = o_[1, 0]
            xs.append(x)
            ys.append(y)
            if c < min_cost:
                min_cost = c
                next_o = o_
                min_cost_action = a

        return min_cost_action, min_cost, next_o

    def greedy_action(self, o, g):
        MAX_SPEED = 1
        u, c, next_o = self.model.act(o, g, MAX_SPEED)
        return u, c, next_o

    @staticmethod
    def state_cost(s, goal):
        return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])
