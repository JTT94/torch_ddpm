from argparse import Namespace
from functorch import vmap
from itertools import repeat


def namespace_to_dict(ns, copy=True):
    d = vars(ns)
    if copy:
        d = d.copy()
    return d


class DataClass(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def items(self):
        return namespace_to_dict(self).items()

    def __getitem__(self, key):
        return self.__getattribute__(key)


batch_mul = vmap(lambda x, y: x * y)


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data
