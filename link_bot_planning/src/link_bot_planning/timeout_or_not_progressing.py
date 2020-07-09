import sys
from time import perf_counter
from typing import Dict

import ompl.base as ob


class TimeoutOrNotProgressing(ob.PlannerTerminationCondition):
    def __init__(self, planner, params: Dict, verbose: int):
        super().__init__(ob.PlannerTerminationConditionFn(self.condition))
        self.params = params
        self.planner = planner
        self.times_called = 1
        self.verbose = verbose
        self.t0 = perf_counter()
        self.timed_out = False
        self.not_progressing = False

    def condition(self):
        self.times_called += 1
        self.not_progressing = self.times_called > self.params['times_called_threshold'] and \
            self.planner.max_distance_from_start < self.params['max_distance_from_start_threshold']
        now = perf_counter()
        dt_s = now - self.t0
        self.timed_out = dt_s > self.params['timeout']
        should_terminate = self.timed_out or self.not_progressing
        if self.verbose >= 3:
            print(f"{self.times_called:6d}, {self.planner.max_distance_from_start:6.4f} {dt_s:7.4f} --> {should_terminate}")
        return should_terminate
