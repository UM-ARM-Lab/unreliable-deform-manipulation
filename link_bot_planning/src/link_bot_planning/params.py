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
    # this is the maximum angular deviation from straight links that we sample in the planner, if using such a sampler
    max_angle_rad: float
    neighborhood_radius: float
    w: float  # for setting up the C space bounds
    h: float
    extent: List[float] = field(init=False)

    def __post_init__(self):
        """
        This assumes the origin of the environment is also the center of the extent
        """
        self.extent = [-self.w / 2, self.w / 2, -self.h / 2, self.h / 2]


@dataclass_json
@dataclass
class SimParams:
    real_time_rate: float
    max_step_size: float
    goal_padding: float
    move_obstacles: bool


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

    w: float = field(init=False)  # for setting up the C space bounds
    h: float = field(init=False)
    extent: List[float] = field(init=False)

    def __post_init__(self):
        """
        This assumes the origin of the environment is also the center of the extent
        """
        self.w = 0 + self.w_cols * self.res
        self.h = 0 + self.h_rows * self.res
        self.extent = [-self.w / 2, self.w / 2, -self.h / 2, self.h / 2]
