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


class NamedTensorResolver(TypeResolveProvider):
    def get_dictionary(self, var):
        return {'shape': str(var.data.shape), 'dtype': var.data.dtype, 'dnames': var.dnames}

    def resolve(self, var, attribute):
        if attribute == 'shape':
            return tuple(var.shape)
        else:
            return getattr(var, attribute, None)

    def can_provide(self, type_object, type_name):
        return type_name == 'NamedTensor'


class NamedTensorToString(StrPresentationProvider):
    def get_str(self, x):
        return f'NT {x.data.dtype} {x.dnames} {x.data.shape} {x.data}'

    def can_provide(self, type_object, type_name):
        return type_name == 'NamedTensor'
