from copy import copy

from moonshine.moonshine_utils import add_batch


class NamedTensor:

    def __init__(self, data, dnames=None):
        self.data = data
        self.dnames = dnames if dnames is not None else []

    def __getitem__(self, args):
        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], str):
            dname = args[0]
            if dname not in self.dnames:
                raise IndexError(f"given dname {dname} is not one of {self.dnames}")
            axis_of_dname = self.dnames.index(dname)
            index = args[1]
            index_slice = self.make_slice(axis_of_dname, index)

            remaining_dnames = copy(self.dnames)
            remaining_dnames.remove(dname)
            return NamedTensor(data=self.data[index_slice], dnames=remaining_dnames)
        else:
            remaining_dnames = copy(self.dnames)
            if isinstance(args, tuple):
                # remove the dname for any dimensions in which a single int index is specified
                # the alternative is that the arg is a slice, in which case we should keep the dname
                dnames_to_remove = []
                for arg_i, arg in enumerate(args):
                    if isinstance(arg, int):
                        # if the dimension that is being index does not have a name, ignore it and break
                        if arg_i >= len(self.dnames):
                            break
                        dnames_to_remove.append(self.dnames[arg_i])
                for dname_to_remove in dnames_to_remove:
                    remaining_dnames.remove(dname_to_remove)
            else:
                remaining_dnames.pop(0)
            return NamedTensor(data=self.data[args], dnames=remaining_dnames)

    def make_slice(self, axis_of_dname, index):
        indexing_args = []
        for j, _ in enumerate(self.data.shape):
            if j == axis_of_dname:
                indexing_args.append(index)
            else:
                indexing_args.append(slice(None, None, None))
        return tuple(indexing_args)

    def shape(self, dname: str):
        axis_of_dname = self.dnames.index(dname)
        return self.data.shape[axis_of_dname]

    def iter(self, dname: str):
        """
        iterates over slices of the axis specified by dname
        """
        axis_of_dname = self.dnames.index(dname)
        for index_along_dname in range(self.data.shape[axis_of_dname]):
            yield self.make_slice(axis_of_dname=axis_of_dname, index=index_along_dname)

    def add_batch(self):
        self.dnames.insert(0, "batch")
        self.data = add_batch(self.data)
        return self

    def remove_batch(self):
        self.dnames.remove("batch")
        self.data = add_batch(self.data)
        return self

    def add_time(self):
        self.dnames.insert(1, "time")
        self.data = add_batch(self.data)
        return self
