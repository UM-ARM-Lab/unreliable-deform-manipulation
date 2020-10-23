from typing import Dict


class BaseDecoderFunction:

    def __init__(self):
        pass

    def decode(self, planned_latent_state: Dict):
        raise NotImplementedError()


class PassThroughDecoderFunction(BaseDecoderFunction):

    def __init__(self):
        super().__init__()

    def decode(self, planned_latent_state: Dict):
        return planned_latent_state
