import argparse
import shutil
from enum import Enum


class ArgsEnum(Enum):

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return cls[s]
        except KeyError:
            raise ValueError()


def my_formatter(prog):
    size = shutil.get_terminal_size((80, 20))
    return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=size.columns, width=size.columns)
