from typing import Optional

from gazebo_msgs.srv import SetPhysicsPropertiesRequest, GetPhysicsPropertiesRequest
from link_bot_pycommon.base_services import BaseServices


class GazeboServices(BaseServices):

    def __init__(self):
        super().__init__()
        self.max_step_size = None

    def setup_env(self, verbose: int, real_time_rate: float, max_step_size: float):
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
