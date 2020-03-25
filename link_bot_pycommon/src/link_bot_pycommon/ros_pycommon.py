from typing import Optional, Dict, List

import numpy as np
import rospy
import std_srvs
from colorama import Fore
from peter_msgs.msg import LinkBotAction
from peter_msgs.srv import ComputeOccupancyRequest, ComputeOccupancy, LinkBotTrajectoryRequest, LinkBotState, WorldControl, \
    LinkBotTrajectory, ExecuteAction, GetObjects, StateSpaceDescription, StateSpaceDescriptionRequest, ExecuteActionRequest

from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest
from ignition.markers import MarkerProvider
from link_bot_pycommon import link_bot_sdf_utils, link_bot_pycommon


class Services:

    def __init__(self):
        self.compute_occupancy = rospy.ServiceProxy('occupancy', ComputeOccupancy)
        self.get_state = rospy.ServiceProxy('link_bot_state', LinkBotState)
        self.execute_action = rospy.ServiceProxy("execute_action", ExecuteAction)
        self.world_control = rospy.ServiceProxy('world_control', WorldControl)
        self.pause = rospy.ServiceProxy('gazebo/pause_physics', std_srvs.srv.Empty)
        self.execute_trajectory = rospy.ServiceProxy("link_bot_execute_trajectory", LinkBotTrajectory)
        self.unpause = rospy.ServiceProxy('gazebo/unpause_physics', std_srvs.srv.Empty)
        self.record = rospy.ServiceProxy('video_recorder', TriggerVideoRecording)
        self.reset = rospy.ServiceProxy("reset", std_srvs.srv.Empty)
        self.get_objects = rospy.ServiceProxy("objects", GetObjects)
        self.states_description = rospy.ServiceProxy("states_description", StateSpaceDescription)
        self.marker_provider = MarkerProvider()

        self.services_to_wait_for = [
            'reset',
            'world_control',
            'link_bot_state',
            'link_bot_execute_trajectory',
            'occupancy',
            'gazebo/pause_physics',
            'gazebo/unpause_physics',
            'gazebo/get_physics_properties',
            'gazebo/set_physics_properties',
        ]

    @staticmethod
    def get_max_speed():
        return rospy.get_param("/link_bot/max_speed")

    @staticmethod
    def get_n_action():
        return rospy.get_param("n_action")

    def get_states_description(self):
        request = StateSpaceDescriptionRequest()
        states_response = self.states_description(request)
        states_dict = {}
        for subspace in states_response.subspaces:
            states_dict[subspace.name] = subspace.dimensions
        return states_dict

    def start_record_trial(self, filename):
        start_msg = TriggerVideoRecordingRequest()
        start_msg.record = True
        start_msg.filename = filename
        start_msg.timeout_in_sec = 300.0
        self.record(start_msg)

    def stop_record_trial(self):
        stop_msg = TriggerVideoRecordingRequest()
        stop_msg.record = False
        self.record(stop_msg)

    def wait(self, verbose):
        if verbose >= 1:
            print(Fore.CYAN + "Waiting for services..." + Fore.RESET)
        for s in self.services_to_wait_for:
            rospy.wait_for_service(s)
        if verbose >= 1:
            print(Fore.CYAN + "Done waiting for services" + Fore.RESET)

    def move_objects(self,
                     max_step_size: float,
                     objects,
                     env_w: float,
                     env_h: float,
                     padding: float,
                     rng: np.random.RandomState):
        pass

    def setup_env(self,
                  verbose: int,
                  real_time_rate: float,
                  reset_gripper_to: Optional,
                  max_step_size: Optional[float] = None):
        pass

    def nudge(self):
        nudge = ExecuteActionRequest()
        nudge.action.gripper1_delta_pos.x = np.random.randn() * 0.1
        nudge.action.gripper1_delta_pos.y = np.random.randn() * 0.1
        self.execute_action(nudge)


def get_occupancy(services,
                  env_w_cols,
                  env_h_rows,
                  res,
                  center_x,
                  center_y):
    request = ComputeOccupancyRequest()
    request.resolution = res
    request.h_rows = env_h_rows
    request.w_cols = env_w_cols
    request.center.x = center_x
    request.center.y = center_y
    request.min_z = 0.05
    request.max_z = 2.00
    request.robot_name = 'link_bot'
    request.request_new = True
    response = services.compute_occupancy(request)
    grid = np.array(response.grid).reshape([response.w_cols, response.h_rows])
    grid = grid.T
    return grid, response


def get_occupancy_data(env_h,
                       env_w,
                       res,
                       services):
    """
    :param env_h:  meters
    :param env_w: meters
    :param res: meters
    :param services: from gazebo_utils
    :return:
    """
    env_h_rows = int(env_h / res)
    env_w_cols = int(env_w / res)
    grid, response = get_occupancy(services,
                                   env_w_cols=env_w_cols,
                                   env_h_rows=env_h_rows,
                                   res=res,
                                   center_x=0,
                                   center_y=0)
    origin = np.array(response.origin)
    full_env_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=res, origin=origin)
    return full_env_data


def get_local_occupancy_data(rows,
                             cols,
                             res,
                             center_point,
                             services):
    """
    :param rows: indices
    :param cols: indices
    :param res: meters
    :param center_point: (x,y) meters
    :param services: from gazebo_utils
    :return: OccupancyData object for local sdf
    """
    grid, response = get_occupancy(services,
                                   env_h_rows=rows,
                                   env_w_cols=cols,
                                   res=res,
                                   center_x=center_point[0],
                                   center_y=center_point[1])
    origin = np.array(response.origin)
    local_occupancy_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=res, origin=origin)
    return local_occupancy_data


def make_trajectory_execution_request(dt, actions):
    req = LinkBotTrajectoryRequest()
    for action in actions:
        action_msg = LinkBotAction()
        action_msg.max_time_per_step = dt
        action_msg.gripper1_delta_pos.x = action[0]
        action_msg.gripper1_delta_pos.y = action[1]
        req.gripper1_traj.append(action_msg)

    return req


def trajectory_execution_response_to_numpy(trajectory_execution_result) -> List[Dict[str, np.ndarray]]:
    actual_path = []
    for objects in trajectory_execution_result.actual_path:
        state = {}
        for object in objects.objects:
            np_config = link_bot_pycommon.flatten_named_points(object.points)
            state[object.name] = np_config
        actual_path.append(state)

    return actual_path


def get_start_states(services, state_keys):
    start_states = {}
    objects_response = services.get_objects()
    for state_key in state_keys:
        for object in objects_response.objects.objects:
            if object.name == state_key:
                state = link_bot_pycommon.flatten_named_points(object.points)
                start_states[state_key] = state

    return start_states
