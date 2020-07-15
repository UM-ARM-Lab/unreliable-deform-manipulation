from typing import Dict


class BaseRecoveryPolicy:

    def __init__(self):
        pass

    def __call__(self, environment: Dict, state: Dict):
        raise NotImplementedError()
