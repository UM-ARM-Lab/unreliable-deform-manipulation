from typing import Tuple, Optional

import numpy as np

from link_bot_pycommon.base_services import Services
from std_srvs.srv import EmptyRequest


class VictorServices(Services):
    def __init__(self):
        super().__init__()

        self.services_to_wait_for.extend([
        ])

    def reset_world(self, verbose, reset_gripper_to: Optional[Tuple[float]] = None):
        empty = EmptyRequest()
        self.reset.call(empty)

    def setup_env(self,
                  verbose: int,
                  real_time_rate: float,
                  reset_gripper_to: Optional,
                  max_step_size: Optional[float] = None):
        self.wait(verbose)
        self.reset_world(verbose, verbose, reset_gripper_to)

    def move_objects(self,
                     max_step_size: float,
                     objects,
                     env_w: float,
                     env_h: float,
                     padding: float,
                     rng: np.random.RandomState):
        pass
