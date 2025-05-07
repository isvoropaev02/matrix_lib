import numpy as np
from scipy.linalg import blas, lu, qr, solve_triangular

from .matrix import Matrix


class SymmetricMatrix(Matrix):

    def __init__(self, data):

        self._size = data.shape[0]
        self._values = np.empty(int(self._size * (self._size + 1) * 0.5))

        # Упаковываем только нижний треугольник
        j = 0
        for i in range(self._size):
            self._values[j : j + i + 1] = data[i, : i + 1]
            j += i + 1

        self._lu_cache = None
        self._qr_cache = None
        self._ldlt_cache = None

    @property
    def shape(self):
        return (self._size, self._size)

    @property
    def dtype(self):
        return self._values[0].dtype

    def __getitem__(self, key):
        row, col = key
        return self._values[
            int(max(row, col) * (max(row, col) + 1) * 0.5 + min(row, col))
        ]

    def __setitem__(self, key, value):
        row, col = key
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
        return SymmetricMatrix(data)

    def to_dense(self):
        mat = np.zeros((self._size, self._size))
        j = 0
        for i in range(self._size):
            mat[i, : i + 1] = self._values[j : j + i + 1]
            j += i + 1
        mat += np.transpose(mat) - mat * np.eye(self._size)
        return mat

    @classmethod
    def zeros(cls, size, default=0):
        new_matrix = np.empty((size, size), dtype=type(default))
        new_matrix[:] = default
        return SymmetricMatrix(new_matrix)

    def __add__(self, other):
        if isinstance(other, SymmetricMatrix):
            assert self._size == other._size, "Размеры матриц не совпадают"
        result = SymmetricMatrix(np.zeros(self.shape))
        result._values += self._values + other._values
        return result

    def __sub__(self, other):
        if isinstance(other, SymmetricMatrix):
            assert self._size == other._size, "Размеры матриц не совпадают"
        result = SymmetricMatrix(np.zeros(self.shape))
        result._values += self._values - other._values
        return result

    def __mul__(self, scalar):
        result = SymmetricMatrix.zeros(self._size)
        result._values += self._values * scalar
        return result

    def __matmul__(self, other):
        if isinstance(other, SymmetricMatrix):
            return blas.dsymm(1.0, self.to_dense(), other.to_dense())
        else:
            return self.to_dense() @ other

    def plu_decomposition(self):

        if self._lu_cache is not None:
            return self._lu_cache
        P, L, U = lu(self.to_dense())
        self._lu_cache = (P, L, U)
        return P, L, U

    def qr_decomposition(self):
        if self._qr_cache is not None:
            return self._qr_cache
        Q, R = qr(self.to_dense())
        self._qr_cache = (Q, R)
        return Q, R

    def ldlt_decomposition(self):
        if self._ldlt_cache is not None:
            return self._ldlt_cache

        L = np.eye(self._size)  # Нижнетреугольная с единичной диагональю
        D = np.zeros(self._size)  # Диагональ

        for j in range(self._size):
            D[j] = self[j, j] - np.sum(L[j, :j] ** 2 * D[:j])

            # Если D[j,j] = 0 → разложение невозможно
            if np.isclose(D[j], 0):
                raise ValueError("LDLᵀ невозможно: нулевой элемент на диагонали D")

            # Вычисляем L[i,j] для i > j
            for i in range(j + 1, self._size):
                L[i, j] = (self[i, j] - np.sum(L[i, :j] * L[j, :j] * D[:j])) / D[j]
        D = np.diag(D)
        self._ldlt_cache = (L, D)
        return L, D

    def det(self):
        if self._ldlt_cache is not None:
            return self._ldlt_cache[1].diagonal().prod()
        if self._qr_cache is not None:
            return self._qr_cache[1].diagonal().prod()

        try:
            _, D = self.ldlt_decomposition()
            det = D.diagonal().prod()
            return det
        except ValueError:
            _, R = self.qr_decomposition()
            det = R.diagonal().prod()
            return det

    def inverse(self):

        if np.issubdtype(self.dtype, np.number):
            if self._ldlt_cache is not None:
                L, D = self._ldlt_cache
            else:
                L, D = self.ldlt_decomposition()

            inv_L = solve_triangular(L, np.eye(self._size, lower=True))
            inv_D = np.diag(1 / np.diag(D))
            return inv_L.T @ inv_D @ inv_L

        raise NotImplementedError("Инверсия реализована только для числовых матриц")

    def solve_slae(self, b):
        try:
            if self._ldlt_cache is not None:
                L, D = self._ldlt_cache
            else:
                L, D = self.ldlt_decomposition()

            y = solve_triangular(L, b, lower=True)  # Ly = b
            z = y / np.diag(D)  # Dz = y
            x = solve_triangular(L.T, z, lower=False)  # Lᵀx = z
            return x

        except ValueError:
            return np.linalg.solve(self.to_dense(), b)

    def __repr__(self):
        # Используем родительский __repr__
        return super().__repr__()
