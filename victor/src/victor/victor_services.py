from typing import Tuple, Optional, Dict

import numpy as np
from std_srvs.srv import EmptyRequest

from link_bot_pycommon.ros_pycommon import Services


class VictorServices(Services):
    def __init__(self):
        super().__init__()

        self.services_to_wait_for.extend([
        ])

    def reset_world(self, verbose, reset_gripper_to: Optional[Tuple[float]] = None):
        empty = EmptyRequest()
        self.reset.call(empty)

    @staticmethod
    def setup_env(verbose: int,
                  real_time_rate: float,
                  reset_gripper_to: Optional,
                  max_step_size: Optional[float] = None,
                  initial_object_dict: Optional[Dict] = None):
        # fire up services
        services = VictorServices()
        services.wait(verbose)

        if reset_gripper_to is not None:
            services.reset_world(verbose, None)

        return services

    def move_objects(self,
                     max_step_size: float,
                     objects,
                     env_w: float,
                     env_h: float,
                     padding: float,
                     rng: np.random.RandomState):
        pass
