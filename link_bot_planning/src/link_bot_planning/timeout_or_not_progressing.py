import sys
from time import perf_counter
from typing import Dict

import ompl.base as ob


class TimeoutOrNotProgressing(ob.PlannerTerminationCondition):
    def __init__(self, planner, params: Dict, verbose: int):
        super().__init__(ob.PlannerTerminationConditionFn(self.condition))
        self.params = params
        self.planner = planner
        self.verbose = verbose
        self.t0 = perf_counter()
        self.attempted_extensions = 0
        self.all_rejected = True
        self.not_progressing = None
        self.timed_out = False

    def condition(self):
        self.not_progressing = self.attempted_extensions >= self.params['attempted_extensions_threshold'] and self.all_rejected
        now = perf_counter()
        dt_s = now - self.t0
        self.timed_out = dt_s > self.params['timeout']
        should_terminate = self.timed_out or self.not_progressing
        if self.verbose >= 3:
            print(f"{self.attempted_extensions:6d}, {self.all_rejected}")
        return should_terminate
