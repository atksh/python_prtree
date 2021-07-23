from .PRTree import _PRTree2D, _PRTree3D

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

    def insert(self, idx, bb):
        if self.n == 0:
            self._tree = self.Klass([idx], [bb])
        else:
            self._tree.insert(idx, bb)

class PRTree3D(PRTree2D):
    Klass = _PRTree3D
