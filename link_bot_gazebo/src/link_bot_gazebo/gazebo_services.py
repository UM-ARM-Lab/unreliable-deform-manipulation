from typing import Optional, Tuple

import numpy as np
import rospy
import std_msgs
from colorama import Fore
from gazebo_msgs.srv import ApplyBodyWrench, SetPhysicsPropertiesRequest, GetPhysicsPropertiesRequest
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyRequest

from link_bot_pycommon.link_bot_pycommon import quaternion_from_euler
from link_bot_pycommon.base_services import Services
from peter_msgs.msg import ModelsPoses, ModelPose
from peter_msgs.srv import WorldControlRequest, ExecuteActionRequest, GetObject, LinkBotReset, \
    LinkBotResetRequest


class GazeboServices(Services):

    def __init__(self):
        super().__init__()
        # we can't mock these
        self.apply_body_wrench = rospy.ServiceProxy('gazebo/apply_body_wrench', ApplyBodyWrench)
        self.link_bot_reset = rospy.ServiceProxy("reset_robot", LinkBotReset)
        self.gazebo_reset = rospy.ServiceProxy("gazebo/reset_world", Empty)

        # don't want to mock these
        self.get_physics = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('gazebo/set_physics_properties', SetPhysicsProperties)

        # not used in real robot experiments
        self.get_tether_state = rospy.ServiceProxy("tether", GetObject)

        # FIXME: Move this kind of interaction with the environment to a separate node, since we will probably
        #  never be able to do it on a real robot
        self.link_bot_mode = rospy.Publisher('link_bot_action_mode', String, queue_size=10)
        self.position_2d_stop = rospy.Publisher('position_2d_stop', std_msgs.msg.Empty, queue_size=10)
        self.position_2d_action = rospy.Publisher('position_2d_action', ModelsPoses, queue_size=10)

    def setup_env(self,
                  verbose: int,
                  real_time_rate: float,
                  reset_robot: Optional,
                  max_step_size: Optional[float] = None,
                  ):
        self.wait(verbose)

        # set up physics
        get = GetPhysicsPropertiesRequest()
        current_physics = self.get_physics.call(get)
        set = SetPhysicsPropertiesRequest()
        set.gravity = current_physics.gravity
        set.ode_config = current_physics.ode_config
        set.max_update_rate = real_time_rate * 1000.0
        if max_step_size is None:
            max_step_size = current_physics.time_step
        set.time_step = max_step_size
        self.set_physics.call(set)

        self.reset_world(verbose, reset_robot)

        # first the controller
        n_action = self.get_n_action()
        stop = ExecuteActionRequest()
        stop.action.action = [0] * n_action
        stop.action.max_time_per_step = 1.0
        self.execute_action(stop)

        self.position_2d_stop.publish(std_msgs.msg.Empty())

    def reset_world(self, verbose, reset_robot: Optional[Tuple[float]] = None):
        empty = EmptyRequest()
        self.reset.call(empty)
        self.gazebo_reset(empty)

        enable_link_bot = String()
        enable_link_bot.data = 'position'
        self.link_bot_mode.publish(enable_link_bot)

        self.reset_robot(reset_robot, verbose)

        if verbose >= 1:
            print(Fore.YELLOW + "World is Reset" + Fore.RESET)

    def reset_robot(self, reset_robot, verbose):
        if reset_robot is not None:
            reset = LinkBotResetRequest()
            reset.point.x = reset_robot[0]
            reset.point.y = reset_robot[1]
            self.link_bot_reset(reset)
            if verbose >= 1:
                print(Fore.YELLOW + "World is Reset" + Fore.RESET)

    @staticmethod
    def random_object_move(model_name: str, w: float, h: float, padding: float, rng: np.random.RandomState):
        move = ModelPose()
        move.pose.position.x = rng.uniform(-w / 2 + padding, w / 2 - padding)
        move.pose.position.y = rng.uniform(-h / 2 + padding, h / 2 - padding)
        q = quaternion_from_euler(0, 0, rng.uniform(-np.pi, np.pi))
        move.pose.orientation.x = q[0]
        move.pose.orientation.y = q[1]
        move.pose.orientation.z = q[2]
        move.pose.orientation.w = q[3]
        move.model_name = model_name
        return move

    def move_objects(self,
                     max_step_size: float,
                     objects,
                     env_w: float,
                     env_h: float,
                     padding: float,
                     rng: np.random.RandomState):
        disable_link_bot = String()
        disable_link_bot.data = 'disabled'

        enable_link_bot = String()
        enable_link_bot.data = 'position'

        # disable the rope controller, enable the objects
        self.link_bot_mode.publish(disable_link_bot)
        # Move the objects
        move_action = ModelsPoses()
        for object_name in objects:
            move = self.random_object_move(object_name, env_w, env_h, padding, rng)
            move_action.actions.append(move)
        self.position_2d_action.publish(move_action)
        # let the move actually occur
        step = WorldControlRequest()
        move_wait_duration = 0.75
        step.steps = int(move_wait_duration / max_step_size)
        self.world_control(step)  # this will block until stepping is complete
        # disable the objects so they stop, enabled the rope controller
        self.position_2d_stop.publish(std_msgs.msg.Empty())
        self.link_bot_mode.publish(enable_link_bot)

        # wait a few steps to ensure the stop message is received
        wait = WorldControlRequest()
        wait.steps = int(2 / max_step_size)
        self.world_control(wait)  # this will block until stepping is complete
