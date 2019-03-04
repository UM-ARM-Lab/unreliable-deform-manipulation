#!/usr/bin/env python
import numpy as np
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
import rospy

from link_bot_agent import a_star
from link_bot_agent import gz_world


def h(v1, v2):
    return np.linalg.norm(np.array(v1.o) - np.array(v2.o))


class LinkConfig:

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None


def get_time_state_action_collision(get_link_state, time, head_vx, head_vy, in_contact):
    state = get_state(get_link_state)
    state.insert(0, time)
    state.insert(1, in_contact)
    state.insert(-1, head_vx)
    state.insert(-1, head_vy)
    return state


def get_time_state_action(get_link_state, time, head_vx, head_vy):
    state = get_state(get_link_state)
    state.insert(0, time)
    state.insert(-1, head_vx)
    state.insert(-1, head_vy)
    return state


def get_state(get_link_state):
    links = {'link_0': LinkConfig(),
             'link_1': LinkConfig(),
             'head': LinkConfig()}

    # get the new states
    for link_name, link_config in links.items():
        req = GetLinkStateRequest()
        req.link_name = link_name
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

    def a_star_plan(self, o, og):
        def mock_sdf(o):
            return False

        graph = gz_world.GzWorldGraph(self.model, mock_sdf)
        planner = a_star.AStar(graph, h)
        shortest_path = planner.shortest_path(gz_world.Vertex(o), gz_world.Vertex(og))

        T = len(shortest_path)
        actions = np.zeros((T, 2, 1))
        os = np.zeros((T, self.M))
        cs = np.zeros(T)
        sbacks = np.zeros((T, self.N))
        for i in range(len(shortest_path)):
            v = shortest_path[i]
            v_ = shortest_path[i + 1]
            s_back = np.linalg.lstsq(self.model.get_A(), v.o, rcond=None)[0]
            sbacks[i] = np.squeeze(s_back)
            os[i] = np.squeeze(v.o)
            cs[i] = self.model.cost(v.o)
            actions[i] = self.model.inverse(v.o, v_.o)

        return actions, cs, os, sbacks

    def greedy_plan(self, o, g, T=1):
        actions = np.zeros((T, 2, 1))
        os = np.zeros((T, self.M))
        cs = np.zeros(T)
        sbacks = np.zeros((T, self.N))

        for i in range(T):
            s_back = np.linalg.lstsq(self.model.get_A(), o, rcond=None)[0]
            sbacks[i] = np.squeeze(s_back)
            os[i] = np.squeeze(o)

            # u, c, next_o = self.sample_action(o, g)
            u, c, next_o = self.greedy_action(o, g)

            cs[i] = c
            actions[i] = u
            o = next_o

        return actions, cs, os, sbacks

    @staticmethod
    def state_cost(s, goal):
        return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])