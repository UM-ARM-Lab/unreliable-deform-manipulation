from typing import Dict, List

import numpy as np

import actionlib
import ros_numpy
import rospy
from arc_utilities.ros_helpers import Listener
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from link_bot_pycommon.grid_utils import extent_array_to_bbox
from link_bot_pycommon.moveit_utils import make_moveit_action_goal
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction
from peter_msgs.srv import GetDualGripperPointsRequest, GetRopeStateRequest
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, SetBoolRequest


class DualArmRopeScenario(DualFloatingGripperRopeScenario):

    def __init__(self):
        super().__init__()
        # TODO: robot_name?
        robot_name = 'victor'
        self.joint_states_listener = Listener(f"{robot_name}/joint_states", JointState)
        self.joint_states_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.goto_home_srv = rospy.ServiceProxy("goto_home", Empty)

        self.service_provider = GazeboServices()

    def reset_robot(self, data_collection_params: Dict):
        if data_collection_params['scene'] == 'tabletop':
            moveit_client = actionlib.SimpleActionClient('move_group', MoveGroupAction)
            moveit_client.wait_for_server()
            joint_names = data_collection_params['home']['name']
            joint_positions = data_collection_params['home']['position']
            goal = make_moveit_action_goal(joint_names, joint_positions)
            moveit_client.send_goal(goal)
            moveit_client.wait_for_result()
            result = moveit_client.get_result()
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                print("Error! code " + str(result.error_code.val))

        elif data_collection_params['scene'] in ['car', 'car2', 'car-floor']:
            positions = np.array(data_collection_params['reset_robot']['position'])
            names = data_collection_params['reset_robot']['name']

            goal = make_moveit_action_goal(names, positions)
            self.move_group_client.send_goal(goal)
            self.move_group_client.wait_for_result()
            result = self.move_group_client.get_result()

            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                rospy.logwarn(f"Failed to reset robot. Running hard reset.")
                self.hard_reset()

    def get_state(self):
        joint_state = self.joint_states_listener.get()
        while True:
            try:
                rope_res = self.get_rope_srv(GetRopeStateRequest())
                break
            except Exception:
                print("CDCPD failed? Restart it!")
                input("press enter.")

        rope_state_vector = []
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        rope_velocity_vector = []
        for v in rope_res.velocities:
            rope_velocity_vector.append(v.x)
            rope_velocity_vector.append(v.y)
            rope_velocity_vector.append(v.z)

        grippers_res = self.get_grippers_srv(GetDualGripperPointsRequest())
        return {
            'gripper1': ros_numpy.numpify(grippers_res.gripper1),
            'gripper2': ros_numpy.numpify(grippers_res.gripper2),
            'link_bot': np.array(rope_state_vector, np.float32),
            'joint_positions': joint_state.position,
            'joint_names': joint_state.name,
        }

    def states_description(self) -> Dict:
        # joints_res = self.joint_states_srv(GetJointStateRequest())
        # FIXME:
        n_joints = 7 + 7 + 14
        return {
            'gripper1': 3,
            'gripper2': 3,
            'link_bot': DualArmRopeScenario.n_links * 3,
            'joint_positions': n_joints
        }

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        super().plot_state_rviz(state, label, **kwargs)
        # TODO: remove this once we no longer need to use the old datasets
        if 'joint_positions' in state and 'joint_names' in state:
            joint_msg = JointState()
            joint_msg.header.stamp = rospy.Time.now()
            joint_msg.position = state['joint_positions']
            if isinstance(state['joint_names'][0], bytes):
                joint_names = [n.decode("utf-8") for n in state['joint_names']]
            elif isinstance(state['joint_names'][0], str):
                joint_names = [str(n) for n in state['joint_names']]
            else:
                raise NotImplementedError(type(state['joint_names'][0]))
            joint_msg.name = joint_names
            self.joint_states_pub.publish(joint_msg)

    def dynamics_dataset_metadata(self):
        joint_state = self.joint_states_listener.get()
        return {
            'joint_names': joint_state.name
        }

    def simple_name(self):
        return "dual_arm"

    def initial_obstacle_poses_with_noise(self, env_rng: np.random.RandomState, obstacles: List):
        raise NotImplementedError()

    def randomize_environment(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        # # move the objects out of the way
        object_reset_poses = {k: (np.ones(3) * 10, np.array([0, 0, 0, 1])) for k in data_collection_params['objects']}
        self.set_object_poses(object_reset_poses)

        # Let go of rope
        release = SetBoolRequest()
        release.data = False
        self.grasping_rope_srv(release)

        # reset rope to home/starting configuration
        self.reset_rope(data_collection_params)
        self.settle()

        # reet robot
        self.reset_robot(data_collection_params)

        # replace the objects in a new random configuration
        if 'scene' not in data_collection_params:
            rospy.logwarn("No scene specified... I assume you want tabletop.")
            random_object_poses = self.random_new_object_poses(env_rng, objects_params)
        elif data_collection_params['scene'] == 'tabletop':
            random_object_poses = self.random_new_object_poses(env_rng, objects_params)
        elif data_collection_params['scene'] == 'car2':
            random_object_poses = self.random_new_object_poses(env_rng, objects_params)
        elif data_collection_params['scene'] == 'car':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        self.set_object_poses(random_object_poses)

        # re-grasp rope
        grasp = SetBoolRequest()
        grasp.data = True
        self.grasping_rope_srv(grasp)

        # wait a second so that the rope can drape on the objects
        self.settle()

        if 'gripper1_action_sample_extent' in data_collection_params:
            gripper1_extent = np.array(data_collection_params['gripper1_action_sample_extent']).reshape([3, 2])
        else:
            gripper1_extent = np.array(data_collection_params['extent']).reshape([3, 2])
        gripper1_bbox_msg = extent_array_to_bbox(gripper1_extent)
        gripper1_bbox_msg.header.frame_id = 'world'
        self.gripper1_bbox_pub.publish(gripper1_bbox_msg)

        if 'gripper2_action_sample_extent' in data_collection_params:
            gripper2_extent = np.array(data_collection_params['gripper2_action_sample_extent']).reshape([3, 2])
        else:
            gripper2_extent = np.array(data_collection_params['extent']).reshape([3, 2])
        gripper2_bbox_msg = extent_array_to_bbox(gripper1_extent)
        gripper2_bbox_msg.header.frame_id = 'world'
        self.gripper2_bbox_pub.publish(gripper2_bbox_msg)

        gripper1_position = env_rng.uniform(gripper1_extent[:, 0], gripper1_extent[:, 1])
        gripper2_position = env_rng.uniform(gripper2_extent[:, 0], gripper2_extent[:, 1])
        return_action = {
            'gripper1_position': gripper1_position,
            'gripper2_position': gripper2_position
        }
        self.execute_action(return_action)
        self.settle()

    def get_environment(self, params: Dict, **kwargs):
        # FIXME: implement
        res = params.get("res", 0.01)
        return get_environment_for_extents_3d(extent=params['extent'],
                                              res=res,
                                              service_provider=self.service_provider,
                                              robot_name=self.robot_name())
