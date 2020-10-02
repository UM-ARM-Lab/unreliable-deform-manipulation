from typing import Dict, List, Optional

import numpy as np

import ros_numpy
import rospy
from arc_utilities.ros_helpers import Listener
from arm_robots.hdt_michigan import Val
from arm_robots.victor import Victor
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from link_bot_pycommon.grid_utils import extent_array_to_bbox
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from peter_msgs.srv import GetDualGripperPointsRequest, GetRopeStateRequest, SetDualGripperPointsRequest, SetDualGripperPoints
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, SetBoolRequest


def get_moveit_robot(robot_namespace: Optional[str] = None):
    if robot_namespace is None:
        robot_namespace = rospy.get_namespace().strip("/")
    if robot_namespace == 'victor':
        return Victor(robot_namespace)
    elif robot_namespace in ['val', 'hdt_michigan']:
        return Val(robot_namespace)
    else:
        raise NotImplementedError(f"robot with namespace {robot_namespace} not implemented")


def attach_or_detach_requests():
    robot_name = rospy.get_namespace().strip("/")

    left_req = AttachRequest()
    left_req.model_name_1 = robot_name
    left_req.link_name_1 = "left_tool_placeholder"
    left_req.model_name_2 = "rope_3d"
    left_req.link_name_2 = "left_gripper"
    left_req.anchor_position.x = 0
    left_req.anchor_position.y = 0
    left_req.anchor_position.z = 0
    left_req.has_anchor_position = True

    right_req = AttachRequest()
    right_req.model_name_1 = robot_name
    right_req.link_name_1 = "right_tool_placeholder"
    right_req.model_name_2 = "rope_3d"
    right_req.link_name_2 = "right_gripper"
    right_req.anchor_position.x = 0
    right_req.anchor_position.y = 0
    right_req.anchor_position.z = 0
    right_req.has_anchor_position = True

    return left_req, right_req


class DualArmRopeScenario(DualFloatingGripperRopeScenario):

    def __init__(self):
        super().__init__()
        self.service_provider = GazeboServices()  # FIXME: won't work on real robot...
        self.joint_states_listener = Listener("joint_states", JointState)
        self.joint_states_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.goto_home_srv = rospy.ServiceProxy("goto_home", Empty)
        self.set_dual_gripper_srv = rospy.ServiceProxy("/rope_3d/set_dual_gripper_points", SetDualGripperPoints)
        self.attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach)
        self.detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)
        self.robot = get_moveit_robot()

    def reset_robot(self, data_collection_params: Dict):
        if data_collection_params['scene'] == 'tabletop':
            self.robot.plan_to_joint_config("both_arms", data_collection_params['home']['position'])
        elif data_collection_params['scene'] in ['car', 'car2', 'car-floor']:
            raise NotImplementedError()

    def get_state(self):
        joint_state = self.joint_states_listener.get(block_until_data=False)
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
            'left_gripper': ros_numpy.numpify(grippers_res.left_gripper),
            'right_gripper': ros_numpy.numpify(grippers_res.right_gripper),
            'link_bot': np.array(rope_state_vector, np.float32),
            'joint_positions': joint_state.position,
            'joint_names': joint_state.name,
        }

    def states_description(self) -> Dict:
        # joints_res = self.joint_states_srv(GetJointStateRequest())
        # FIXME:
        n_joints = 7 + 7 + 14
        return {
            'left_gripper': 3,
            'right_gripper': 3,
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

    def on_before_data_collection(self):
        self.service_provider.pause()
        self.move_rope_to_match_grippers()
        self.attach_rope_to_grippers()
        self.service_provider.play()
        left_gripper_position = np.array([-0.2, 0.5, 0.3])
        right_gripper_position = np.array([0.2, -0.5, 0.3])
        init_action = {
            'left_gripper_position': left_gripper_position,
            'right_gripper_position': right_gripper_position,
            'speed': 0.25,
        }
        self.execute_action(init_action)

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

        if 'left_gripper_action_sample_extent' in data_collection_params:
            left_gripper_extent = np.array(data_collection_params['left_gripper_action_sample_extent']).reshape([3, 2])
        else:
            left_gripper_extent = np.array(data_collection_params['extent']).reshape([3, 2])
        left_gripper_bbox_msg = extent_array_to_bbox(left_gripper_extent)
        left_gripper_bbox_msg.header.frame_id = 'world'
        self.left_gripper_bbox_pub.publish(left_gripper_bbox_msg)

        if 'right_gripper_action_sample_extent' in data_collection_params:
            right_gripper_extent = np.array(data_collection_params['right_gripper_action_sample_extent']).reshape([3, 2])
        else:
            right_gripper_extent = np.array(data_collection_params['extent']).reshape([3, 2])
        right_gripper_bbox_msg = extent_array_to_bbox(left_gripper_extent)
        right_gripper_bbox_msg.header.frame_id = 'world'
        self.right_gripper_bbox_pub.publish(right_gripper_bbox_msg)

        left_gripper_position = env_rng.uniform(left_gripper_extent[:, 0], left_gripper_extent[:, 1])
        right_gripper_position = env_rng.uniform(right_gripper_extent[:, 0], right_gripper_extent[:, 1])
        return_action = {
            'left_gripper_position': left_gripper_position,
            'right_gripper_position': right_gripper_position
        }
        self.execute_action(return_action)
        self.settle()

    def execute_action(self, action: Dict):
        speed = action['speed']
        left_gripper_points = [action['left_gripper_position']]
        right_gripper_points = [action['right_gripper_position']]
        tool_names = ["left_tool_placeholder", "right_tool_placeholder"]
        grippers = [left_gripper_points, right_gripper_points]
        self.robot.follow_jacobian_to_position("both_arms", tool_names, grippers, speed)

    def get_environment(self, params: Dict, **kwargs):
        # FIXME: implement
        res = params.get("res", 0.01)
        return get_environment_for_extents_3d(extent=params['extent'],
                                              res=res,
                                              service_provider=self.service_provider,
                                              robot_name=self.robot_name())

    @staticmethod
    def robot_name():
        raise NotImplementedError()

    def attach_rope_to_grippers(self):
        left_req, right_req = attach_or_detach_requests()
        self.attach_srv(left_req)
        self.attach_srv(right_req)

    def detach_rope_to_grippers(self):
        left_req, right_req = attach_or_detach_requests()
        self.detach_srv(left_req)
        self.detach_srv(right_req)

    def move_rope_to_match_grippers(self):
        left_transform = self.tf.get_transform("robot_root", "left_tool_placeholder")
        right_transform = self.tf.get_transform("robot_root", "right_tool_placeholder")
        move = SetDualGripperPointsRequest()
        move.left_gripper.x = left_transform[0, 3]
        move.left_gripper.y = left_transform[1, 3]
        move.left_gripper.z = left_transform[2, 3]
        move.right_gripper.x = right_transform[0, 3]
        move.right_gripper.y = right_transform[1, 3]
        move.right_gripper.z = right_transform[2, 3]
        self.set_dual_gripper_srv(move)
