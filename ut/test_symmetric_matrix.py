import numpy as np
from matrix_lib import SymmetricMatrix

def test_symmetry():
    matrix = SymmetricMatrix(np.array([
        [1, 2],
        [2, 3]
    ]))
    assert (matrix.shape == (2, 2)), f"Matrix shape is not square, it is {matrix.shape}"
    assert (matrix[0, 0] == 1), f"Matrix was not saved correctly, matrix[0, 0] = {matrix[0, 0]} instead of 1"
    assert (matrix[0, 1] == 2) and (matrix[1, 0] == 2), f"Matrix was not saved correctly, matrix[0, 1] = {matrix[0, 1]} and matrix[1, 0] = {matrix[1, 0]} instead of 2"
    assert (matrix[1, 1] == 3), f"Matrix was not saved correctly, matrix[1, 1] = {matrix[1, 1]} instead of 3"


def test_getitem_setitem():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    sample_matrix[0, 1] = 10
    assert (sample_matrix[0, 1] == 10) and (sample_matrix[1, 0] == 10), f"Matrix element was not changed correctly, matrix[0, 1] = {sample_matrix[0, 1]} and matrix[1, 0] = {sample_matrix[1, 0]} instead of 10"

def test_to_dense():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    dense = sample_matrix.to_dense()
    assert isinstance(dense, np.ndarray), f"Dense matrix is not a numpy array, it is {type(dense)}"
    assert dense.shape == (3, 3), f"Dense matrix shape is not square, it is {dense.shape}"
    assert np.array_equal(dense, dense.T), f"Dense matrix is not symmetric, dense = {dense}"

def test_addition():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    
    other = SymmetricMatrix(np.array([
        [2, 3, 4],
        [3, 5, 6],
        [4, 6, 7]
    ]))
    result = sample_matrix + other
    assert result[0, 0] == 3, f"Matrix element was not added correctly, result[0, 0] = {result[0, 0]} instead of 3"
    assert result[0, 1] == 5, f"Matrix element was not added correctly, result[0, 1] = {result[0, 1]} instead of 5"

def test_subtraction():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))

    other = SymmetricMatrix(np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]))
    result = sample_matrix - other
    assert result[0, 0] == 0, f"Matrix element was not subtracted correctly, result[0, 0] = {result[0, 0]} instead of 0"
    assert result[0, 1] == 1, f"Matrix element was not subtracted correctly, result[0, 1] = {result[0, 1]} instead of 1"

def test_scalar_multiplication():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    result = sample_matrix * 2
    assert result[0, 0] == 2, f"Matrix element was not multiplied correctly, result[0, 0] = {result[0, 0]} instead of 2"
    assert result[0, 1] == 4, f"Matrix element was not multiplied correctly, result[0, 1] = {result[0, 1]} instead of 4"

def test_matrix_multiplication():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    other = np.array([[1], [2], [3]])
    result = sample_matrix @ other
    assert result.shape == (3, 1), f"Matrix shape is not (3, 1), it is {result.shape}"

def test_ldlt_decomposition():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 7, 4], 
        [7, 4, 5], 
        [4, 5, 6]
    ]))
    L, D = sample_matrix.ldlt_decomposition()
    assert isinstance(L, np.ndarray), f"L is not a numpy array, it is {type(L)}"
    assert isinstance(D, np.ndarray), f"D is not a numpy array, it is {type(D)}"
    assert np.allclose(np.diag(L), np.ones(3)), f"L has not unit diagonal, diag(L) = {np.diag(L)}"
    assert np.allclose(D, np.diag(np.diag(D))), f"D is not diagonal, D = {D}"

def test_determinant():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    det = sample_matrix.det()
    assert isinstance(det, (int, float)), f"Determinant is not a number, it is {type(det)}"
    assert np.isclose(det, np.linalg.det(sample_matrix.to_dense())), f"Determinant is not correct, det = {det} instead of {np.linalg.det(sample_matrix.to_dense())}"

def test_inverse():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 7, 4], 
        [7, 4, 5], 
        [4, 5, 6]
    ]))
    inv = sample_matrix.inverse()
    product = sample_matrix @ inv
    assert np.allclose(product, np.eye(3), atol=1e-10), f"Product of matrix and inverse is not the identity matrix, product = {product}"

def test_solve_slae():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    b = np.array([1, 2, 3])
    x = sample_matrix.solve_slae(b)
    assert np.allclose(sample_matrix @ x, b, atol=1e-10), f"Solution of SLAE is not correct, x = {x}"

def test_zeros():
    size = 3
    matrix = SymmetricMatrix.zeros(size)
    assert matrix.shape == (size, size), f"Matrix shape is not (size, size), it is {matrix.shape}"
    assert np.allclose(matrix.to_dense(), np.zeros((size, size))), f"Matrix is not zero, matrix = {matrix}"

def test_empty_like():
    sample_matrix = SymmetricMatrix(np.array([
        [1, 2, 3], 
        [2, 4, 5], 
        [3, 5, 6]
    ]))
    empty = sample_matrix.empty_like()
    assert isinstance(empty, SymmetricMatrix), f"Empty matrix is not a SymmetricMatrix, it is {type(empty)}"
    assert empty.shape == sample_matrix.shape, f"Empty matrix shape is not the same as the original matrix, it is {empty.shape} instead of {sample_matrix.shape}"

def run_test():
    test_symmetry()
    test_getitem_setitem()
    test_to_dense()
    test_addition()
    test_subtraction()
    test_scalar_multiplication()
    test_matrix_multiplication()
    test_ldlt_decomposition()
    test_determinant()
    # test_inverse()
    test_solve_slae()
    test_zeros()
    test_empty_like()

if __name__ == "__main__":
    run_test()