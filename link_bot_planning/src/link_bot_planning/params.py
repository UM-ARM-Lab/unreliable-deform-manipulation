from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json


# TODO: make external method for turning on and off visualization options (ros param server?, RQT GUI?)
@dataclass_json
@dataclass
class PlannerParams:
    timeout: float
    max_v: float
    goal_threshold: float
    random_epsilon: float
    max_angle_rad: float  # this is the maximum angular deviation from straight links that we sample in the planner


@dataclass_json
@dataclass
class EnvParams:
    w: float
    h: float
    real_time_rate: float
    max_step_size: float
    goal_padding: float
    extent: List[float] = field(init=False)
    move_obstacles: bool

    def __post_init__(self):
        """
        This assumes the origin of the environment is also the center of the extent
        """
        self.extent = [-self.w / 2, self.w / 2, -self.h / 2, self.h / 2]


@dataclass_json
@dataclass
class LocalEnvParams:
    h_rows: int
    w_cols: int
    res: float


@dataclass_json
@dataclass
class FullEnvParams:
    h_rows: int
    w_cols: int
    res: float
