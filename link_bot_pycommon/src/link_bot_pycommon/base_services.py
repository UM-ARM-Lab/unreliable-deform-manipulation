from typing import Optional, Dict

from colorama import Fore

import rospy
import std_srvs
from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from geometry_msgs.msg import Pose
from ignition.markers import MarkerProvider
from peter_msgs.srv import ComputeOccupancy, WorldControl, GetObjects, \
    StateSpaceDescription, StateSpaceDescriptionRequest, ActionSpaceDescription, \
    ActionSpaceDescriptionRequest


class Services:

    def __init__(self):
        self.compute_occupancy = rospy.ServiceProxy('occupancy', ComputeOccupancy)
        self.world_control = rospy.ServiceProxy('world_control', WorldControl)
        self.pause = rospy.ServiceProxy('gazebo/pause_physics', std_srvs.srv.Empty)
        self.unpause = rospy.ServiceProxy('gazebo/unpause_physics', std_srvs.srv.Empty)
        self.record = rospy.ServiceProxy('video_recorder', TriggerVideoRecording)
        self.reset = rospy.ServiceProxy("reset", std_srvs.srv.Empty)
        self.action_description = rospy.ServiceProxy("link_bot/actions", ActionSpaceDescription)
        self.get_objects = rospy.ServiceProxy("objects", GetObjects)
        self.states_description = rospy.ServiceProxy("states_description", StateSpaceDescription)
        self.marker_provider = MarkerProvider()
        self.get_physics = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('gazebo/set_physics_properties', SetPhysicsProperties)

        self.services_to_wait_for = [
            'reset',
            'states_description',
            'world_control',
            'occupancy',
            'gazebo/pause_physics',
            'gazebo/unpause_physics',
            'gazebo/get_physics_properties',
            'gazebo/set_physics_properties',
        ]

    def get_action_description(self):
        req = ActionSpaceDescriptionRequest()
        res = self.action_description(req)
        actions_dict = {}
        for subspace in res.subspaces:
            actions_dict[subspace.name] = {
                'dim': subspace.dimensions,
                'lower_bounds': subspace.lower_bounds,
                'upper_bounds': subspace.upper_bounds,
            }
        return actions_dict

    def move_objects(self, object_moves: Dict[str, Pose]):
        pass

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
            if verbose >= 3:
                print("Waiting for {}".format(s))
            rospy.wait_for_service(s)
        if verbose >= 1:
            print(Fore.CYAN + "Done waiting for services" + Fore.RESET)

    def setup_env(self, verbose: int, real_time_rate: float, max_step_size: float):
        raise NotImplementedError()

    def reset_world(self, verbose, reset_robot: Optional = None):
        raise NotImplementedError()

    def get_movable_object_positions(self, movable_obstacles):
        pass
