import numpy as np
from _pydevd_bundle.pydevd_extension_api import StrPresentationProvider, TypeResolveProvider
from colorama import Fore


class TfTensorResolver(TypeResolveProvider):
    def get_dictionary(self, var):
        return {'shape': str(var.shape), 'dtype': var.dtype.name}

    def resolve(self, var, attribute):
        if attribute == 'shape':
            return tuple(var.shape)
        else:
            return getattr(var, attribute, None)

    def can_provide(self, type_object, type_name):
        return type_name in ['EagerTensor', 'Tensor']


class TfTensorToString(StrPresentationProvider):
    def get_str(self, x):
        x_np = x.numpy()
        return f'{x.dtype.name} {x.shape} {np.array2string(x_np)}'

    def can_provide(self, type_object, type_name):
        return type_name in ['EagerTensor', 'Tensor']
