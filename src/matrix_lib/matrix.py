import numpy as np
from scipy.linalg import blas, lu, qr, solve_triangular

from .text_block import TextBlock


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

        text = [
            [TextBlock.from_str(f"{self[r, c]}") for c in range(self.width)]
            for r in range(self.height)
        ]
        width_el = np.array(
            list(map(lambda row: list(map(lambda el: el.width, row)), text))
        )
        height_el = np.array(
            list(map(lambda row: list(map(lambda el: el.height, row)), text))
        )
        width_column = np.max(width_el, axis=0)
        width_total = np.sum(width_column)
        height_row = np.max(height_el, axis=1)
        result = []
        for r in range(self.height):
            lines = TextBlock.merge(
                text[r][c].format(width=width_column[c], height=height_row[r])
                for c in range(self.width)
            )
            for t in lines:
                result.append(f"| {t} |")
            if (
                len(lines) > 0
                and len(lines[0]) > 0
                and lines[0][0] == "|"
                and r < self.height - 1
            ):
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

            assert (
                self.width == other.width and self.height == other.height
            ), f"Shapes does not match: {self.shape} != {other.shape}"

            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] + other[r, c]
            return matrix
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):

            assert (
                self.width == other.width and self.height == other.height
            ), f"Shapes does not match: {self.shape} != {other.shape}"

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

            assert (
                self.width == other.height
            ), f"Shapes does not match: {self.shape} != {other.shape}"

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
        # if isinstance(element, Fraction):
        #     return 1 / element
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
        new_matrix = np.empty((size, size), dtype=type(default))
        new_matrix[:] = default
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
