import numpy as np
from scipy.linalg import lu, inv, qr

from .text_block import *

class Matrix:
    """Общий предок для всех матриц."""

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def __repr__(self):
        """Возвращает текстовое представление для матрицы."""
        text = [[TextBlock.from_str(f"{self[r, c]}") for c in range(self.width)] for r in range(self.height)]
        width_el = np.array(list(map(lambda row: list(map(lambda el: el.width, row)), text)))
        height_el = np.array(list(map(lambda row: list(map(lambda el: el.height, row)), text)))
        width_column = np.max(width_el, axis=0)
        width_total = np.sum(width_column)
        height_row = np.max(height_el, axis=1)
        result = []
        for r in range(self.height):
            lines = TextBlock.merge(
                text[r][c].format(width=width_column[c], height=height_row[r]) for c in range(self.width))
            for l in lines:
                result.append(f"| {l} |")
            if len(lines) > 0 and len(lines[0]) > 0 and lines[0][0] == '|' and r < self.height - 1:
                result.append(f'| {" " * (width_total + self.width)}|')
        return "\n".join(result)

    def empty_like(self, width=None, height=None):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] + other[r, c]
            return matrix
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] - other[r, c]
            return matrix
        return NotImplemented

    def __mul__(self, other):
        return self.__matmul__(other)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(other.width):
                    acc = None
                    for k in range(self.width):
                        add = self[r, k] * other[k, c]
                        acc = add if acc is None else acc + add
                    matrix[r, c] = acc
            return matrix
        return NotImplemented

    def inverse(self):
        raise NotImplementedError

    def invert_element(self, element):
        if isinstance(element, float):
            return 1 / element
        if isinstance(element, Fraction):
            return 1 / element
        if isinstance(element, Matrix):
            return element.inverse()
        raise TypeError


class FullMatrix(Matrix):
    """
    Заполненная матрица с элементами произвольного типа.
    """

    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        return FullMatrix(data)

    @classmethod
    def zero(_cls, height, width, default=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return FullMatrix(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        row, column = key
        return self.data[row, column]

    def __setitem__(self, key, value):
        row, column = key
        self.data[row, column] = value


class SymmetricMatrix(Matrix):

    def __init__(self, data):

        self._size = data.shape[0]
        self._values = []

        # Упаковываем только нижний треугольник
        for i in range(self._size):
            self._values.append(data[i, :i + 1])

        self._lu_cache = None
        self._qr_cache = None

    @property
    def shape(self):
        return (self._size, self._size)

    @property
    def dtype(self):
        return self._values[0][0].dtype

    def __getitem__(self, key):
        row, col = key
        return self._values[max(row, col)][min(row, col)]

    def __setitem__(self, key, value):
        row, col = key
        self._values[max(row, col)][min(row, col)] = value

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
        for i in range(self._size):
            mat[i, :i + 1] = self._values[i]
            mat[:i, i] = self._values[i][:-1]
        return mat

    @classmethod
    def zeros(cls, size, dtype=np.float64):
        new_matrix = cls.__new__(cls)
        new_matrix._size = size
        new_matrix._values = [np.zeros(i + 1, dtype=dtype) for i in range(size)]
        return new_matrix

    def __add__(self, other):
        if isinstance(other, SymmetricMatrix):
            assert self._size == other._size, "Размеры матриц не совпадают"
        result = SymmetricMatrix(np.zeros(self.shape))
        for i in range(self._size):
            result._values[i] = self._values[i] + other._values[i]
        return result

    def __sub__(self, other):
        if isinstance(other, SymmetricMatrix):
            assert self._size == other._size, "Размеры матриц не совпадают"
        result = SymmetricMatrix(np.zeros(self.shape))
        for i in range(self._size):
            result._values[i] = self._values[i] - other._values[i]
        return result

    def __mul__(self, scalar):
        result = SymmetricMatrix.zeros(self._size, dtype=self.dtype)
        for i in range(self._size):
            result._values[i] = self._values[i] * scalar
        return result

    def __matmul__(self, other):
        if isinstance(other, SymmetricMatrix):
            result = SymmetricMatrix.zeros(self._size)
            for i in range(self._size):
                for j in range(i, self._size):
                    total = 0.0
                    for k in range(self._size):
                        a_ik = self._values[max(i, k)][min(i, k)] if k != i else self._values[i][i]
                        b_kj = other._values[max(k, j)][min(k, j)] if k != j else other._values[j][j]
                        total += a_ik * b_kj
                    result._values[j][i] = total
            return result
        else:
            return self.to_dense() @ other

    def lu_decomposition(self):

        if self._lu_cache is not None:
            return self._lu_cache
        # TODO: можно как то использоваьб LDL^T разложение для полож. опр. матриц

        #     n = self._size
        #     L = np.eye(n)
        #     D = np.zeros(n)
        #
        #     for j in range(n):
        #         D[j] = self._values[j][j] - sum(L[j,k]**2 * D[k] for k in range(j))
        #
        #         for i in range(j+1, n):
        #             L[i,j] = (self._values[i][j] - sum(L[i,k]*L[j,k]*D[k] for k in range(j))) / D[j]
        #
        #     ldlt = True
        #     self._lu_cache = (L, D, ldlt)
        #     return L, D, ldlt

        P, L, U = lu(self.to_dense())
        self._lu_cache = (P, L, U)
        return P, L, U

    def qr_decomposition(self):
        if self._qr_cache is not None:
            return self._qr_cache
        Q, R = qr(self.to_dense())
        self._qr_cache = (Q, R)
        return Q, R

    def det(self):

        _, R = self.qr_decomposition()
        det = R.diagonal().prod()
        return det

    def inverse(self):

        if np.issubdtype(self.dtype, np.number):
            inv_data = inv(self.to_dense())
            return SymmetricMatrix(inv_data)
        raise NotImplementedError("Инверсия реализована только для числовых матриц")

    def __repr__(self):
        # Используем родительский __repr__
        return super().__repr__()
