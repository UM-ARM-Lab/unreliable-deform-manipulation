from typing import Optional

import numpy as np
import rospy
import std_srvs
from colorama import Fore

from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest
from ignition.markers import MarkerProvider
from peter_msgs.srv import ComputeOccupancy, LinkBotState, ExecuteAction, WorldControl, LinkBotTrajectory, GetObjects, \
    StateSpaceDescription, StateSpaceDescriptionRequest, ExecuteActionRequest


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
            'states_description',
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
        return rospy.get_param("link_bot/max_speed")

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