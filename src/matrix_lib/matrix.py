import numpy as np

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
