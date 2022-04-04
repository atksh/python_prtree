import codecs
import pickle

from .PRTree import _PRTree2D, _PRTree3D, _PRTree4D

__all__ = [
    "PRTree2D",
    "PRTree3D",
    "PRTree4D",
]


def dumps(obj):
    if obj is None:
        return None
    else:
        return pickle.dumps(obj)


def loads(obj):
    if obj is None:
        return None
    else:
        return pickle.loads(obj)


class PRTree2D:
    Klass = _PRTree2D

    def __init__(self, *args, **kwargs):
        self._tree = self.Klass(*args, **kwargs)

    def __getattr__(self, name):
        def handler_function(*args, **kwargs):
            ret = getattr(self._tree, name)(*args, **kwargs)
            return ret

        return handler_function

    @property
    def n(self):
        return self._tree.size()

    def __len__(self):
        return self.n

    def erase(self, idx):
        if self.n == 0:
            raise ValueError("Nothing to erase")
        self._tree.erase(idx)

    def set_obj(self, idx, obj):
        objdumps = dumps(obj)
        self._tree.set_obj(idx, objdumps)

    def get_obj(self, idx):
        obj = self._tree.get_obj(idx)
        return loads(obj)

    def insert(self, idx=None, bb=None, obj=None):
        if idx is None and obj is None:
            raise ValueError("Specify index or obj")
        if idx is None:
            idx = self.n + 1
        if bb is None:
            raise ValueError("Specify bounding box")

        objdumps = dumps(obj)
        if self.n == 0:
            self._tree = self.Klass([idx], [bb])
            self._tree.set_obj(idx, objdumps)
        else:
            self._tree.insert(idx, bb, objdumps)

    def query(self, *args, return_obj=False):
        if len(args) == 1:
            out = self._tree.query(*args)
        else:
            out = self._tree.query(args)
        if return_obj:
            objs = [self.get_obj(i) for i in out]
            return objs
        else:
            return out


class PRTree3D(PRTree2D):
    Klass = _PRTree3D


class PRTree4D(PRTree2D):
    Klass = _PRTree4D
