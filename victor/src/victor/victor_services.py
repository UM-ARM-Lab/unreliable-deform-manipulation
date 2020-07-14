from typing import Tuple, Optional

import numpy as np

from link_bot_pycommon.base_services import BaseServices
from std_srvs.srv import EmptyRequest


class VictorServices(BaseServices):
    def __init__(self):
        super().__init__()

    def setup_env(self, verbose: int, real_time_rate: float, max_step_size: float):
        pass
