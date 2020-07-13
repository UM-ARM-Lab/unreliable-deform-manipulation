from typing import Tuple, Optional

import numpy as np

from link_bot_pycommon.base_services import BaseServices
from std_srvs.srv import EmptyRequest


class VictorServices(BaseServices):
    def __init__(self):
        super().__init__()
