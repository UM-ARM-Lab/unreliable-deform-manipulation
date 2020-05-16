from typing import Optional, List, Dict

import numpy as np
from colorama import Fore

import rospy
from gazebo_msgs.srv import ApplyBodyWrench, SetPhysicsPropertiesRequest, GetPhysicsPropertiesRequest
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from geometry_msgs.msg import Pose
from link_bot_pycommon.base_services import Services
from link_bot_pycommon.link_bot_pycommon import quaternion_from_euler
from peter_msgs.msg import ModelsPoses
from peter_msgs.srv import WorldControlRequest, ExecuteActionRequest, GetObject, LinkBotReset, \
    LinkBotResetRequest, Position2DEnable, Position2DEnableRequest, Position2DAction, Position2DActionRequest
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyRequest


class GazeboServices(Services):

    def __init__(self, movable_object_names):
        super().__init__()
        # we can't mock these
        self.max_step_size = None
        self.apply_body_wrench = rospy.ServiceProxy('gazebo/apply_body_wrench', ApplyBodyWrench)
        self.reset_robot_service = rospy.ServiceProxy("reset_robot", LinkBotReset)
        self.gazebo_reset = rospy.ServiceProxy("gazebo/reset_world", Empty)

        # don't want to mock these
        self.get_physics = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('gazebo/set_physics_properties', SetPhysicsProperties)

        # not used in real robot experiments
        self.get_tether_state = rospy.ServiceProxy("tether", GetObject)

        # TODO: Consider moving this kind of interaction with the environment to a separate node, since we will probably
        #  never be able to do it on a real robot
        self.link_bot_mode = rospy.Publisher('link_bot_action_mode', String, queue_size=10)
        self.movable_object_names = movable_object_names
        self.movable_object_services = {}
        for object_name in self.movable_object_names:
            self.movable_object_services[object_name] = {
                'enable': rospy.ServiceProxy(f'{object_name}/position_2d_enable', Position2DEnable),
                'action': rospy.ServiceProxy(f'{object_name}/position_2d_action', Position2DAction),
                'stop': rospy.ServiceProxy(f'{object_name}/position_2d_stop', Empty),
            }

    def setup_env(self,
                  verbose: int,
                  real_time_rate: float,
                  reset_robot: Optional,
                  reset_world: Optional[bool] = True,
                  stop: Optional[bool] = True,
                  max_step_size: Optional[float] = None,
                  ):
        self.wait(verbose)

        # set up physics
        get_physics_msg = GetPhysicsPropertiesRequest()
        current_physics = self.get_physics.call(get_physics_msg)
        set_physics_msg = SetPhysicsPropertiesRequest()
        set_physics_msg.gravity = current_physics.gravity
        set_physics_msg.ode_config = current_physics.ode_config
        set_physics_msg.max_update_rate = real_time_rate * 1000.0
        if max_step_size is None:
            max_step_size = current_physics.time_step
        self.max_step_size = max_step_size
        set_physics_msg.time_step = max_step_size
        self.set_physics.call(set_physics_msg)

        if reset_world:
            self.reset_world(verbose, reset_robot)

        if stop:
            n_action = self.get_n_action()
            stop = ExecuteActionRequest()
            stop.action.action = [0] * n_action
            stop.action.max_time_per_step = 1.0
            self.execute_action(stop)

        for movable_object_services in self.movable_object_services.values():
            movable_object_services['stop'](EmptyRequest())

    def reset_world(self, verbose, reset_robot: Optional = None):
        empty = EmptyRequest()
        self.reset.call(empty)
        self.gazebo_reset(empty)

        enable_link_bot = String()
        enable_link_bot.data = 'position'
        self.link_bot_mode.publish(enable_link_bot)

        self.reset_robot(verbose, reset_robot)

        if verbose >= 1:
            print(Fore.YELLOW + "World is Reset" + Fore.RESET)

    def reset_robot(self, verbose: int, reset_robot: Optional = None):
        if reset_robot is not None:
            reset = LinkBotResetRequest()
            reset.point.x = reset_robot[0]
            reset.point.y = reset_robot[1]
            self.reset_robot_service(reset)
            if verbose >= 1:
                print(Fore.YELLOW + "World is Reset" + Fore.RESET)

    @staticmethod
    def random_object_position(w: float, h: float, padding: float, rng: np.random.RandomState) -> Pose:
        pose = Pose()
        pose.position.x = rng.uniform(-w / 2 + padding, w / 2 - padding)
        pose.position.y = rng.uniform(-h / 2 + padding, h / 2 - padding)
        q = quaternion_from_euler(0, 0, rng.uniform(-np.pi, np.pi))
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def random_move_objects(self,
                            objects: List[str],
                            env_w: float,
                            env_h: float,
                            padding: float,
                            rng: np.random.RandomState):
        random_object_positions = {name: self.random_object_position(w=env_w, h=env_h, padding=padding, rng=rng)
                                   for name in objects}
        self.move_objects(random_object_positions)

    def move_objects(self, object_moves: Dict[str, Pose]):
        disable_link_bot = String()
        disable_link_bot.data = 'disabled'

        enable_link_bot = String()
        enable_link_bot.data = 'position'

        # disable the rope controller, enable the objects
        self.link_bot_mode.publish(disable_link_bot)
        for object_name, pose in object_moves.items():
            movable_object_services = self.movable_object_services[object_name]
            enable_req = Position2DEnableRequest()
            enable_req.enable = True
            movable_object_services['enable'](enable_req)

        # Move the objects
        move_action = ModelsPoses()
        for object_name, pose in object_moves.items():
            movable_object_services = self.movable_object_services[object_name]
            move_action_req = Position2DActionRequest()
            move_action_req.pose = pose
            movable_object_services['action'](move_action_req)
        # let the move actually occur
        step = WorldControlRequest()
        move_wait_duration = 5.00
        step.steps = int(move_wait_duration / self.max_step_size)
        self.world_control(step)  # this will block until stepping is complete

        # stop the objects, enabled the rope controller
        for object_name, pose in object_moves.items():
            movable_object_services = self.movable_object_services[object_name]
            movable_object_services['stop'](EmptyRequest())
        self.link_bot_mode.publish(enable_link_bot)

        # wait a few steps to ensure the stop message is received
        wait = WorldControlRequest()
        wait.steps = int(2 / self.max_step_size)
        self.world_control(wait)  # this will block until stepping is complete
