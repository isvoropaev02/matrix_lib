import numpy as np
from scipy.linalg import blas

from .matrix import Matrix


class TriangularMatrix(Matrix):
    """Инициализируется 2 параметра: 1) сама матрица data, 2) параметр transp = False если матрица нижнетреугольная
    и равен True если матрица верхнетреугольная."""

    def __init__(self, data, transp=False):

        self._size = data.shape[0]
        self._values = np.empty(int(self._size * (self._size + 1) * 0.5))
        self._transp = transp
        if not transp:  # Упаковываем только нижний треугольник
            j = 0
            for i in range(self._size):
                self._values[j : j + i + 1] = data[i, : i + 1]
                j += i + 1

        else:  # транспонированием переводим в нижнетреугольную матрицу и устанавливаем флаг в self._transp
            data = np.transpose(data)
            j = 0
            for i in range(self._size):
                self._values[j : j + i + 1] = data[i, : i + 1]
                j += i + 1
            self._transp = True

        self._lu_cache = None
        self._qr_cache = None

    @property
    def shape(self):
        return (self._size, self._size)

    @property
    def dtype(self):
        return self._values[0].dtype

    def __getitem__(self, key):
        row, col = key
        if (not self._transp and col > row) or (self._transp and col < row):
            return 0
        return self._values[
            int(max(row, col) * (max(row, col) + 1) * 0.5 + min(row, col))
        ]

    def __setitem__(self, key, value):
        row, col = key
        assert not (
            (not self._transp and col > row) or (self._transp and col < row)
        ), "Матрица выйдет из класса треугольных"
        self._values[int(max(row, col) * (max(row, col) + 1) * 0.5 + min(row, col))] = (
            value
        )

    def empty_like(self, width=None, height=None):
        dtype = self.dtype
        if width is None:
            width = self._size
        if height is None:
            height = self._size
        data = np.empty((height, width), dtype=dtype)
        return TriangularMatrix(data, self._transp)

    def to_dense(self):
        mat = np.zeros((self._size, self._size))
        j = 0
        for i in range(self._size):
            mat[i, : i + 1] = self._values[j : j + i + 1]
            j += i + 1
        if self._transp:
            mat = np.transpose(mat)
        return mat

    @classmethod
    def zeros(cls, size, default=0):
        new_matrix = np.zeros((size, size), dtype=type(default))
        return TriangularMatrix(new_matrix)

    def __add__(self, other):
        if isinstance(other, TriangularMatrix):
            assert self._size == other._size, "Размеры матриц не совпадают"
        if self._transp == other._transp:
            result = TriangularMatrix(np.zeros(self.shape), self._transp)
            result._values += self._values + other._values
            return result
        else:
            return self.to_dense() + other.to_dense()

    def __sub__(self, other):
        if isinstance(other, TriangularMatrix):
            assert self._size == other._size, "Размеры матриц не совпадают"
        if self._transp == other._transp:
            result = TriangularMatrix(np.zeros(self.shape), self._transp)
            result._values += self._values - other._values
            return result
        else:
            return self.to_dense() - other.to_dense()

    def __mul__(self, scalar):
        result = TriangularMatrix.zeros(self._size)
        result._values += self._values * scalar
        result._transp = self._transp
        return result

    def __matmul__(self, other):
        if isinstance(other, TriangularMatrix):
            if (not self._transp) and (not other._transp):
                return TriangularMatrix(
                    blas.dtrmm(
                        1.0,
                        self.to_dense(),
                        other.to_dense(),
                        side=0,
                        lower=True,
                        trans_a=0,
                    ),
                    False,
                )
            elif self._transp and other._transp:
                return TriangularMatrix(
                    blas.dtrmm(
                        1.0,
                        self.to_dense(),
                        other.to_dense(),
                        side=0,
                        lower=False,
                        trans_a=0,
                    ),
                    True,
                )
            elif (not self._transp) and other._transp:
                return blas.dtrmm(
                    1.0,
                    self.to_dense(),
                    other.to_dense(),
                    side=0,
                    lower=True,
                    trans_a=0,
                    overwrite_b=False,
                )
            else:
                return blas.dtrmm(
                    1.0,
                    self.to_dense(),
                    other.to_dense(),
                    side=0,
                    lower=False,
                    trans_a=0,
                    overwrite_b=False,
                )
        else:
            return self.to_dense() @ other

    def det(self):
        return np.prod(
            [self._values[int(i * (i + 1) * 0.5 - 1)] for i in range(self._size)]
        )

    def inverse(self):
        if not self._transp:

            n = self._size
            inv_L = np.zeros((self._size, self._size))

            for i in range(n):
                inv_L[i, i] = 1 / self[i, i]
                for j in range(i):
                    inv_L[i, j] = (
                        -np.sum(
                            np.fromiter(
                                (self[i, k] * inv_L[k, j] for k in range(j, i)),
                                dtype=np.float64,
                            )
                        )
                        / self[i, i]
                    )

            return inv_L

        else:
            n = self._size
            inv_U = np.zeros((self._size, self._size))

            for i in reversed(range(n)):  # Идём снизу вверх
                inv_U[i, i] = 1 / self[i, i]  # Диагональ

                for j in range(i + 1, n):
                    inv_U[i, j] = (
                        -np.sum(
                            np.fromiter(
                                (self[i, k] * inv_U[k, j] for k in range(i + 1, j + 1)),
                                dtype=np.float64,
                            )
                        )
                        / self[i, i]
                    )

            return inv_U

    def solve_slae(self, b):

        n = self._size
        x = np.zeros_like(b, dtype=float)
        if not self._transp:
            for i in range(n):
                x[i] = (
                    b[i]
                    - np.sum(
                        np.fromiter(
                            (self[i, k] * x[k] for k in range(i)), dtype=np.float64
                        )
                    )
                ) / self[i, i]

            return x
        else:

            for i in reversed(range(n)):
                x[i] = (
                    b[i]
                    - np.sum(
                        np.fromiter(
                            (self[i, k] * x[k] for k in range(i + 1, n)),
                            dtype=np.float64,
                        )
                    )
                ) / self[i, i]

            return x

    def __repr__(self):
        # Используем родительский __repr__
        return super().__repr__()
