import numpy as np
from scipy.linalg import solve_banded, qr

from .matrix import Matrix


class BandMatrix(Matrix):
    def __init__(self, data, lower_bandwidth=None, upper_bandwidth=None):
        """
        Инициализация ленточной матрицы.

        Параметры:
            data: Входная матрица или 2D массив
            lower_bandwidth: Число поддиагоналей (kl)
            upper_bandwidth: Число наддиагоналей (ku)

        NB: Сложение и вычитание доступно ТОЛЬКО для матриц одинаковой структуры (с одинаковыми  размерами, kl и ku)
        """
        self._size = data.shape[0]

        # Автоматическое определение ширины ленты, если не задано
        if lower_bandwidth is None or upper_bandwidth is None:
            non_zero = np.nonzero(data)
            row_indices = non_zero[0]
            col_indices = non_zero[1]

            # Вычисляем максимальное расстояние от диагонали
            diag_dist = np.abs(row_indices - col_indices)

            if lower_bandwidth is None:
                lower_bandwidth = np.max(diag_dist[row_indices > col_indices]) if np.any(
                    row_indices > col_indices) else 0

            if upper_bandwidth is None:
                upper_bandwidth = np.max(diag_dist[col_indices > row_indices]) if np.any(
                    col_indices > row_indices) else 0

        self.kl = lower_bandwidth
        self.ku = upper_bandwidth
        self._values = self._pack_banded(data)
        self._qr_cache = None

    def _pack_banded(self, data):
        """Упаковка матрицы в ленточный формат хранения."""
        n = self._size
        packed = np.zeros((self.kl + self.ku + 1, n), dtype=data.dtype)

        for i in range(n):
            for j in range(max(0, i - self.kl), min(n, i + self.ku + 1)):
                packed[self.ku + i - j, j] = data[i, j]

        return packed

    def _unpack_banded(self):
        """Распаковка ленточной матрицы в плотный формат."""
        n = self._size
        dense = np.zeros((n, n), dtype=self._values.dtype)

        for j in range(n):
            for i in range(max(0, j - self.ku), min(n, j + self.kl + 1)):
                dense[i, j] = self._values[self.ku + i - j, j]

        return dense

    @property
    def shape(self):
        return (self._size, self._size)

    @property
    def dtype(self):
        return self._values.dtype

    def __getitem__(self, key):
        row, col = key
        # Проверка границ
        if not (0 <= row < self._size and 0 <= col < self._size):
            raise IndexError("Индекс выходит за границы матрицы")

        # Проверка на нахождение вне ленты

        if col <= row and abs(row - col) > self.kl:
            return 0  # Элементы вне ленты считаем нулями
        elif col > row and abs(row - col) > self.ku:
            return 0  # Элементы вне ленты считаем нулями
        # Доступ к элементам внутри ленты
        return self._values[self.ku + row - col, col]

    def __setitem__(self, key, value):
        row, col = key
        if not (0 <= row < self._size and 0 <= col < self._size):
            raise IndexError("Индекс выходит за границы матрицы")
        if abs(row - col) > self.kl + self.ku:
            raise ValueError(f"Нельзя установить элемент ({row},{col}) вне ленты")
        self._values[self.ku + row - col, col] = value

    def empty_like(self, width=None, height=None):
        if width is None:
            width = self._size
        if height is None:
            height = self._size
        data = np.empty((height, width), dtype=self.dtype)
        return BandMatrix(data, self.kl, self.ku)

    def to_dense(self):
        return self._unpack_banded()

    @classmethod
    def zeros(cls, size, kl, ku, default=0):
        data = BandMatrix(np.zeros((size, size), dtype=type(default)), kl, ku)
        data._values[:] = default
        return data

    def __add__(self, other):
        if isinstance(other, BandMatrix):
            assert self._size == other._size, "Размеры матриц должны совпадать"
            assert self.kl == other.kl and self.ku == other.ku, "Ширины лент должны совпадать"
            result = BandMatrix.zeros(self._size, self.kl, self.ku)
            result._values = self._values + other._values
            return result
        else:
            return self.to_dense() + other

    def __sub__(self, other):
        if isinstance(other, BandMatrix):
            assert self._size == other._size, "Размеры матриц должны совпадать"
            assert self.kl == other.kl and self.ku == other.ku, "Ширины лент должны совпадать"
            result = BandMatrix.zeros(self._size, self.kl, self.ku)
            result._values = self._values - other._values
            return result
        else:
            return self.to_dense() - other

    def __mul__(self, scalar):
        result = BandMatrix.zeros(self._size, self.kl, self.ku)
        result._values = self._values * scalar
        return result

    def __matmul__(self, other):
        if isinstance(other, BandMatrix):
            res_kl = min(self.kl + other.kl, self._size - 1)
            res_ku = min(self.ku + other.ku, self._size - 1)
            result = BandMatrix.zeros(self._size, res_kl, res_ku)

            for i in range(self._size):
                start_j = max(0, i - self.kl)
                end_j = min(self._size, i + self.ku + 1)

                for j in range(max(0, i - res_kl), min(self._size, i + res_ku + 1)):
                    start_k = max(start_j, j - other.ku)
                    end_k = min(end_j, j + other.kl + 1)

                    s = 0.0
                    for k in range(start_k, end_k):
                        s += self[i, k] * other[k, j]
                    result[i, j] = s

            return result
        else:
            return self.to_dense() @ other

    def qr_decomposition(self):

        if self._qr_cache is not None:
            return self._qr_cache

        Q, R = qr(self.to_dense())

        self._qr_cache = (Q, R)
        return Q, R

    def solve_slae(self, b):
        try:
            # Сначала пробуем использовать ленточный решатель
            return solve_banded((self.kl, self.ku), self._values, b)
        except ValueError:
            # Если не получилось, используем плотный решатель
            return np.linalg.solve(self.to_dense(), b)

    def det(self):

        if self._qr_cache is not None:
            return self._qr_cache[1].diagonal().prod()

        _, R = self.qr_decomposition()
        det = R.diagonal().prod()
        return det

    def inverse(self):
        """Вычисление обратной матрицы решением СЛАУ для каждого базисного вектора."""
        if np.issubdtype(self.dtype, np.number):
            inv = np.zeros((self._size, self._size))
            for i in range(self._size):
                e = np.zeros(self._size)
                e[i] = 1
                inv[:, i] = self.solve_slae(e)
            return inv
        raise NotImplementedError("Обращение реализовано только для числовых матриц")

    def __repr__(self):
        # Используем родительский __repr__
        return super().__repr__()

