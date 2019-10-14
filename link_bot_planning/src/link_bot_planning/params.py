from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json


# TODO: make external method for turning on and off visualization options (ros param server?, RQT GUI?)
@dataclass_json
@dataclass
class PlannerParams:
    timeout: float
    max_v: float


@dataclass_json
@dataclass
class EnvParams:
    w: float
    h: float
    real_time_rate: float
    goal_padding: float
    extent: List[float] = field(init=False)

    def __post_init__(self):
        """
        This assumes the origin of the environment is also the center of the extent
        """
        self.extent = [-self.w / 2, self.w / 2, -self.h / 2, self.h / 2]


@dataclass_json
@dataclass
class SDFParams:
    full_h_m: float
    full_w_m: float
    local_h_rows: int
    local_w_cols: int
    res: float
