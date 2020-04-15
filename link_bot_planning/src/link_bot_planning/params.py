from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class PlannerParams:
    max_v: float
    goal_threshold: float
    random_epsilon: float
    w: float  # for setting up the C space bounds
    h: float
    extent: List[float] = field(init=False)

    timeout: float
    sampler_type: str

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
    movable_obstacles: Optional[List[str]]
    nudge: Optional[bool] = True


@dataclass_json
@dataclass
class LocalEnvParams:
    h_rows: int
    w_cols: int


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


@dataclass_json
@dataclass
class CollectDynamicsParams:
    res: float
    full_env_h_m: float
    full_env_w_m: float
    max_step_size: float
    dt: float
    steps_per_traj: int
    movable_obstacles: List[str]
    move_objects_every_n: int
    trajs_per_file: int
    no_obstacles: bool
    goal_radius_m: Optional[float] = None
    goal_h_m: Optional[float] = None
    goal_w_m: Optional[float] = None
    reset_robot: Optional[List[float]] = None
    reset_world: Optional[bool] = None
