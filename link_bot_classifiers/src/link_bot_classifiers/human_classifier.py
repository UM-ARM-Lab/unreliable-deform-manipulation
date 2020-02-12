from time import time
from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier
import matplotlib.pyplot as plt
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_pycommon import link_bot_sdf_utils


class ManualClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()

    def predict(self, local_env_data: List[link_bot_sdf_utils.OccupancyData], s1: np.ndarray, s2: np.ndarray,
                action: np.ndarray) -> List[float]:
        # assumes 1
        local_env_data = local_env_data[0]

        manual_label = None
        fig = plt.figure()
        ax = plt.gca()

        t0 = None

        def keypress_label(event):
            nonlocal manual_label
            dt = time() - t0
            if dt < 1:
                # be patient, good labeler!
                return

            if event.key == 'y':
                plt.close(fig)
                manual_label = 1.0
            if event.key == 'n':
                plt.close(fig)
                manual_label = 0.0

        t0 = time()
        plot_classifier_data(planned_env=local_env_data.data,
                             planned_env_extent=local_env_data.extent,
                             planned_state=s1[0],
                             planned_next_state=s2[0],
                             planned_env_origin=local_env_data.origin,
                             action=action[0, 0],
                             ax=ax)
        fig.canvas.mpl_connect('key_press_event', keypress_label)
        plt.show(block=True)

        return [manual_label]
