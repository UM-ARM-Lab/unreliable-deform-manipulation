from typing import Union, List


class BatchedDictionary:

    def __getitem__(self, index: Union[slice, int, List[int], str]):
        if isinstance(index, slice):
            pass
        elif isinstance(index, int):
            pass
        elif isinstance(index, list):
            pass

        elif isinstance(index, str):
            pass
