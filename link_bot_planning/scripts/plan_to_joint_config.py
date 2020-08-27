import argparse
import json

import actionlib
import rospy

from link_bot_pycommon.moveit_utils import make_moveit_action_goal
from moveit_msgs.msg import MoveGroupAction, MoveItErrorCodes


def main():
    rospy.init_node("plan_to_joint_config")

    parser = argparse.ArgumentParser()
    parser.add_argument("configs")
    parser.add_argument("name", type=str)
    args = parser.parse_args()

    client = actionlib.SimpleActionClient('move_group', MoveGroupAction)
    client.wait_for_server()

    with open(args.configs, "r") as configs_file:
        configs = json.load(configs_file)

    config = configs[args.name]
    positions = config['position']
    names = config['name']

    goal = make_moveit_action_goal(names, positions)
    client.send_goal(goal)
    client.wait_for_result()
    result = client.get_result()
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
        print("Error! code " + str(result.error_code.val))
    else:
        print("Success!")


if __name__ == "__main__":
    main()
