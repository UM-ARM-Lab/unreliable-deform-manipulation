import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.latex_utils import make_cell
from link_bot_pycommon.matplotlib_utils import save_unconstrained_layout
from link_bot_pycommon.metric_utils import row_stats


class ResultsMetric:
    def __init__(self, args, analysis_params: Dict, results_dir: pathlib.Path, name: str):
        self.args = args
        self.params = analysis_params
        self.results_dir = results_dir
        self.name = name
        self.values = {}
        self.method_indices = {}

        self.fig, self.ax = plt.subplots(figsize=(16, 10))

    def setup_method(self, method_name: str, metadata: Dict):
        self.values[method_name] = []

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        raise NotImplementedError()

    def aggregate_trial(self, method_name: str, scenario: ExperimentScenario, trial_datum: Dict):
        self.values[method_name].append(self.get_metric(scenario, trial_datum))

    def convert_to_numpy_arrays(self):
        for k, v in self.values.items():
            self.values[k] = np.array(v)

    def make_table(self, table_format):
        table_data = []
        for method_name, values_for_method in self.values.items():
            table_data.append(self.make_row(method_name, values_for_method, table_format))
        return self.get_table_header(), table_data

    def get_table_header(self):
        raise NotImplementedError()

    def make_row(self, method_name: str, values_for_method: np.array, table_format: str):
        row = [
            make_cell(method_name, table_format),
        ]
        row.extend(row_stats(values_for_method))
        return row

    def make_figure(self):
        # Methods need to have consistent colors across different plots
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for (method_name, values_for_method), color in zip(self.values.items(), colors):
            self.add_to_figure(method_name=method_name, values=values_for_method, color=color)

    def add_to_figure(self, method_name: str, values: List, color):
        raise NotImplementedError()

    def finish_figure(self):
        self.ax.legend()

    def save_figure(self):
        save_unconstrained_layout(self.fig, self.results_dir / (self.name + ".png"))

    def enumerate_methods(self):
        for i, k in enumerate(self.values):
            self.method_indices[k] = i


class BoxplotOverTrialsPerMethod(ResultsMetric):
    def __init__(self, args, analysis_params: Dict, results_dir: pathlib.Path, name: str):
        super().__init__(args, analysis_params, results_dir, name)
        self.ax.set_xlabel("Method")
        self.ax.set_ylabel(self.name)

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return trial_datum['total_time']

    def add_to_figure(self, values: List, method_name: str, color):
        x = self.method_indices[method_name]
        self.ax.boxplot(values,
                        positions=[x],
                        patch_artist=True,
                        boxprops=dict(facecolor=color, color=color),
                        capprops=dict(color=color),
                        whiskerprops=dict(color=color),
                        medianprops=dict(color=color))

    def get_table_header(self):
        return ["Name", self.name]

    def finish_figure(self):
        # don't a legend for these plots
        readable_keys = [k.replace("_", " ") for k in self.values.keys()]
        self.ax.set_xticklabels(readable_keys)


class FinalExecutionToGoalError(ResultsMetric):
    def __init__(self, args, analysis_params: Dict, results_dir: pathlib.Path):
        super().__init__(args, analysis_params, results_dir, "final execution to goal distance")
        max_error = self.params["max_error"]
        self.errors_thresholds = np.linspace(0.01, max_error, self.params["n_error_bins"])
        self.ax.set_xlabel("Task Error Threshold")
        self.ax.set_ylabel("Success Rate")
        self.ax.set_ylim([-0.1, 100.5])
        self.goal_threshold = None

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        goal = trial_datum['goal']
        final_actual_state = trial_datum['end_state']
        final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
        return final_execution_to_goal_error

    def setup_method(self, method_name: str, metadata: Dict):
        super().setup_method(method_name, metadata)
        planner_params = metadata['planner_params']
        self.goal_threshold = planner_params['goal_threshold']

    def add_to_figure(self, method_name: str, values: List, color):
        success_rate_at_thresholds = []
        for threshold in self.errors_thresholds:
            success_rate_at_threshold = np.count_nonzero(values < threshold) / len(values) * 100
            success_rate_at_thresholds.append(success_rate_at_threshold)
        readable_method_name = method_name.replace("_", " ")
        self.ax.plot(self.errors_thresholds, success_rate_at_thresholds, label=readable_method_name, color=color)
        self.ax.axvline(self.goal_threshold, color='k', linestyle='--')

    def get_table_header(self):
        return ["Name", "Dynamics", "Classifier", "min", "max", "mean", "median", "std"]

    def make_row(self, method_name: str, values_for_method: np.array, table_format: str):
        row = [
            make_cell(method_name, table_format),
            # make_cell(table_config["dynamics"], tablefmt),
            # make_cell(table_config["classifier"], tablefmt),
        ]
        row.extend(row_stats(values_for_method))
        return row


class NRecoveryActions(BoxplotOverTrialsPerMethod):
    def __init__(self, args, analysis_params: Dict, results_dir: pathlib.Path):
        super().__init__(args, analysis_params, results_dir, "Recovery Actions")

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        steps = trial_datum['steps']
        n_recovery = 0
        for step in steps:
            if step['type'] == 'executed_recovery':
                n_recovery += 1
        return n_recovery


class NPlanningAttempts(BoxplotOverTrialsPerMethod):
    def __init__(self, args, analysis_params: Dict, results_dir: pathlib.Path):
        super().__init__(args, analysis_params, results_dir, "Planning Attempts")

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return len(trial_datum['steps'])


class TotalTime(BoxplotOverTrialsPerMethod):
    def __init__(self, args, analysis_params: Dict, results_dir: pathlib.Path):
        super().__init__(args, analysis_params, results_dir, "Total Time")
        self.ax.set_ylabel("Total Time")

    def get_metric(self, scenario: ExperimentScenario, trial_datum: Dict):
        return trial_datum['total_time']
