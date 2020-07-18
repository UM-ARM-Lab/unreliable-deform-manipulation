#!/usr/bin/env python
import rospy


from peter_msgs.srv import WorldControl, WorldControlRequest, SetRopeState, SetRopeStateRequest, SetBoolRequest, SetBool

if __name__ == "__main__":
    rospy.init_node("move_rope")
    grasping_rope_srv = rospy.ServiceProxy("set_grasping_rope", SetBool)
    set_rope_state_srv = rospy.ServiceProxy("set_rope_state", SetRopeState)
    world_control_srv = rospy.ServiceProxy("world_control", WorldControl)

    def reset_rope():
        reset = SetRopeStateRequest()

        reset.gripper1.x = 1.5
        reset.gripper1.y = -0.2
        reset.gripper1.z = 1.0
        reset.gripper2.x = 1.5
        reset.gripper2.y = 0.2
        reset.gripper2.z = 1.0

        set_rope_state_srv(reset)

    def settle():
        req = WorldControlRequest()
        req.seconds = 6
        world_control_srv(req)

    # Let go of rope
    release = SetBoolRequest()
    release.data = False
    grasping_rope_srv(release)

    # reset rope to home/starting configuration
    reset_rope()
    settle()

    k = raw_input("re-grasp rope? [Y/n]")
    if k != 'n':
        # re-grasp rope
        grasp = SetBoolRequest()
        grasp.data = True
        grasping_rope_srv(grasp)
        settle()
