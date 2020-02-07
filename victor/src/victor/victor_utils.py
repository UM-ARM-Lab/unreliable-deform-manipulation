import rospy

from link_bot_pycommon.ros_pycommon import Services


class VictorServices(Services):
    def __init__(self):
        super().__init__()

        self.get_rope = rospy.ServiceProxy('/cdcpd/tracked_points', None)

        self.services_to_wait_for.extend([
            '/cdcpd/tracked_points'
        ])
