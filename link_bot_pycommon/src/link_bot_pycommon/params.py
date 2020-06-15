from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Environment:
    env: np.ndarray
    extent: np.array
    origin: np.array
    res: float


@dataclass_json
@dataclass
class CollectDynamicsParams:
    extent: np.array
    res: float
    max_step_size: float
    dt: float
    steps_per_traj: int
    movable_obstacles: Dict[str, Dict[str, List[float]]]
    trajs_per_file: int
    no_obstacles: bool
    goal_extent: np.array
    goal_radius_m: Optional[float]
    reset_robot: Optional[List[float]]
    reset_world: Optional[bool]
