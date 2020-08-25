import roslaunch

from gazebo_msgs.srv import SetPhysicsPropertiesRequest, GetPhysicsPropertiesRequest
from link_bot_pycommon.base_services import BaseServices


class GazeboServices(BaseServices):

    def __init__(self):
        super().__init__()
        self.max_step_size = None
        self.gazebo_process = None

    def launch(self, params):
        world_filename = params['world_filename']
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        roslaunch_args = ['link_bot_gazebo', 'gazebo.launch', f'world:={world_filename}']

        roslaunch_file1 = roslaunch.rlutil.resolve_launch_arguments(roslaunch_args)
        roslaunch_args1 = roslaunch_args[2:]

        launch_files = [(roslaunch_file1, roslaunch_args1)]

        self.gazebo_process = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
        self.gazebo_process.start()

    def kill(self):
        self.gazebo_process.terminate()

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
