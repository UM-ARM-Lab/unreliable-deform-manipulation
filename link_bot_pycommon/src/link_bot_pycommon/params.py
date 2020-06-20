from dataclasses import dataclass
from typing import List, Dict

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
    movable_objects: Dict[str, Dict[str, List[float]]]
    trajs_per_file: int
    no_objects: bool
    settling_time: float
    min_dist_between_grippers: float
    max_dist_between_grippers: float
