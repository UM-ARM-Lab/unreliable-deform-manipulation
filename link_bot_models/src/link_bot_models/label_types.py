import numpy as np

from link_bot_pycommon import link_bot_pycommon


class LabelType(link_bot_pycommon.ArgsEnum):
    SDF = 0
    Overstretching = 1

    @staticmethod
    def mask(label_types):
        mask = np.zeros(len(LabelType), dtype=np.int)
        for label_type in label_types:
            i = label_type.value
            mask[i] = 1
        return mask
