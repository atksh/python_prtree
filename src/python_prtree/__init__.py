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
            # Handle empty tree cases for methods that cause segfaults
            if self.n == 0 and name in ('rebuild', 'save'):
                # These operations are not meaningful/safe on empty trees
                if name == 'rebuild':
                    return  # No-op for empty tree
                elif name == 'save':
                    raise ValueError("Cannot save empty tree")

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

        # Handle erasing the last element (library limitation workaround)
        if self.n == 1:
            # Call underlying erase to validate index, then handle the library bug
            try:
                self._tree.erase(idx)
                # If we get here, erase succeeded (shouldn't happen with n==1)
                return
            except RuntimeError as e:
                error_msg = str(e)
                if "Given index is not found" in error_msg:
                    # Index doesn't exist - re-raise the error
                    raise
                elif "#roots is not 1" in error_msg:
                    # This is the library bug we're working around
                    # Index was valid, so recreate empty tree
                    self._tree = self.Klass()
                    return
                else:
                    # Some other RuntimeError - re-raise it
                    raise

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
        # Handle empty tree case to prevent segfault
        if self.n == 0:
            return []

        if len(args) == 1:
            out = self._tree.query(*args)
        else:
            out = self._tree.query(args)
        if return_obj:
            objs = [self.get_obj(i) for i in out]
            return objs
        else:
            return out

    def batch_query(self, queries, *args, **kwargs):
        # Handle empty tree case to prevent segfault
        if self.n == 0:
            # Return empty list for each query
            import numpy as np
            if hasattr(queries, 'shape'):
                return [[] for _ in range(len(queries))]
            return []

        return self._tree.batch_query(queries, *args, **kwargs)


class PRTree3D(PRTree2D):
    Klass = _PRTree3D


class PRTree4D(PRTree2D):
    Klass = _PRTree4D
