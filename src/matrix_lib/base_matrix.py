import numpy as np


class MatrixBase:
    def __init__(self, m, n):
        self._mat = np.empty((m, n), dtype=np.float32)

    @property
    def n(self):
        return self._mat.shape[1]

    @property
    def m(self):
        return self._mat.shape[0]

    @classmethod
    def zeros(cls, m, n):
        a = cls(m, n)
        a._mat.fill(0)
        return a
