import numpy as np
from matrix_lib import TriangularMatrix

def test_initialization():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    assert np.allclose(matrix.to_dense(), np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])), f"Matrix is not initialized correctly, it is {matrix.to_dense()} instead of [[1, 0, 0], [2, 3, 0], [4, 5, 6]]"

def test_shape():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    assert matrix.shape == (3, 3), f"Matrix shape is not (3, 3), it is {matrix.shape}"

def test_dtype():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    assert matrix.dtype == np.float64, f"Matrix dtype is not np.float64, it is {matrix.dtype}"

def test_getitem():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    assert matrix[0, 0] == 1, f"Matrix element is not 1, it is {matrix[0, 0]}"
    assert matrix[0, 1] == 0, f"Matrix element is not 0, it is {matrix[0, 1]}"
    assert matrix[0, 2] == 0, f"Matrix element is not 0, it is {matrix[0, 2]}"

def test_setitem():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    matrix[0, 0] = 10
    assert matrix[0, 0] == 10, f"Matrix element is not 10, it is {matrix[0, 0]}"

def test_empty_like():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    empty = matrix.empty_like(3, 3)
    assert empty.shape == (3, 3), f"Shape is not (3, 3), it is {empty.shape}"
    assert empty.dtype == matrix.dtype, f"dtype is not {matrix.dtype}, it is {empty.dtype}"
    assert empty._transp == matrix._transp, f"transp is not {matrix._transp}, it is {empty._transp}"

def test_to_dense():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    assert np.allclose(matrix.to_dense(), np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])), f"Matrix is not dense, it is {matrix.to_dense()} instead of [[1, 0, 0], [2, 3, 0], [4, 5, 6]]"

def test_zeros():
    matrix = TriangularMatrix.zeros(3, 3)
    assert np.allclose(matrix.to_dense(), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])), f"Matrix is not dense, it is {matrix.to_dense()} instead of [[0, 0, 0], [0, 0, 0], [0, 0, 0]]"

def test_addition():
    matrix1 = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    matrix2 = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    result = matrix1 + matrix2
    assert np.allclose(result.to_dense(), np.array([[2, 0, 0], [4, 6, 0], [8, 10, 12]])), f"Matrix is not dense, it is {result.to_dense()} instead of [[2, 0, 0], [4, 6, 0], [8, 10, 12]]"

def test_subtraction():
    matrix1 = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    matrix2 = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    result = matrix1 - matrix2
    assert np.allclose(result.to_dense(), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])), f"Matrix is not dense, it is {result.to_dense()} instead of [[0, 0, 0], [0, 0, 0], [0, 0, 0]]"

def test_scalar_multiplication():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    result = matrix * 2
    assert np.allclose(result.to_dense(), np.array([[2, 0, 0], [4, 6, 0], [8, 10, 12]])), f"Matrix is not dense, it is {result.to_dense()} instead of [[2, 0, 0], [4, 6, 0], [8, 10, 12]]"

def test_matrix_multiplication():
    matrix1 = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    matrix2 = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    result = matrix1 @ matrix2
    assert np.allclose(result.to_dense(), np.array([[1, 0, 0], [8, 9, 0], [38, 45, 36]])), f"Matrix is not dense, it is {result.to_dense()} instead of [[1, 0, 0], [8, 9, 0], [24, 33, 36]]"

def test_det():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    assert matrix.det() == 18, f"Matrix determinant is not 18, it is {matrix.det()}"

def test_inverse():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    assert np.allclose(matrix.inverse(), np.array([[1, 0, 0], [-0.66666667, 0.33333333, 0], [-0.11111111, -0.27777778, 0.16666667]])), f"Matrix is not dense, it is {matrix.inverse()} instead of [[1, 0, 0], [-0.66666667, 0.33333333, 0], [-0.11111111, -0.27777778, 0.16666667]]"

def test_solve_slae():
    matrix = TriangularMatrix(np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), transp=False)
    b = np.array([1, 2, 3])
    x = matrix.solve_slae(b)
    assert np.allclose(x, np.array([ 1, 0, -0.16666667])), f"Matrix is not dense, it is {x} instead of [1, 0, -0.16666667]"

def run_test():
    test_initialization()
    test_shape()
    test_dtype()
    test_getitem()
    test_setitem()
    test_empty_like()
    test_to_dense()
    test_zeros()
    test_addition()
    test_subtraction()
    test_scalar_multiplication()
    test_matrix_multiplication()
    test_det()
    test_inverse()
    test_solve_slae()


if __name__ == "__main__":
    run_test()














