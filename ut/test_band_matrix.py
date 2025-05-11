import numpy as np
from matrix_lib import BandMatrix

def test_band_matrix():
    band_matrix = BandMatrix(np.array([[1, 2, 0], 
                                       [4, 5, 6], 
                                       [0, 8, 9]]), 
                                        lower_bandwidth=1, 
                                        upper_bandwidth=1)
    assert band_matrix.shape == (3, 3), f"Shape is not (3, 3), it is {band_matrix.shape}"
    assert band_matrix.kl == 1, f"kl is not 1, it is {band_matrix.kl}"
    assert band_matrix.ku == 1, f"ku is not 1, it is {band_matrix.ku}"
    assert np.allclose(band_matrix.to_dense(), np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]])), f"Data is not correct, it is {band_matrix.to_dense()}"

def test_band_matrix_addition():
    band_matrix1 = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    band_matrix2 = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    result = band_matrix1 + band_matrix2
    assert np.allclose(result.to_dense(), np.array([[2, 4, 0], [8, 10, 12], [0, 16, 18]])), f"Result is not correct, it is {result.to_dense()} instead of [[2, 4, 0], [8, 10, 12], [0, 16, 18]]"

def test_band_matrix_subtraction():
    band_matrix1 = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    band_matrix2 = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    result = band_matrix1 - band_matrix2
    assert np.allclose(result.to_dense(), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])), f"Result is not correct, it is {result.to_dense()} instead of [[0, 0, 0], [0, 0, 0], [0, 0, 0]]"

def test_band_matrix_multiplication():
    band_matrix = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    result = band_matrix * 2
    assert np.allclose(result.to_dense(), np.array([[2, 4, 0], [8, 10, 12], [0, 16, 18]])), f"Result is not correct, it is {result.to_dense()} instead of [[2, 4, 0], [8, 10, 12], [0, 16, 18]]"

def test_band_matrix_matrix_multiplication():
    band_matrix = BandMatrix(np.array([[1, 2, 0], 
                                       [4, 5, 6], 
                                       [0, 8, 9]]), 
                                       lower_bandwidth=1, 
                                       upper_bandwidth=1)
    matrix = np.array([[1, 2, 3], 
                       [4, 5, 6], 
                       [7, 8, 9]])
    result = band_matrix @ matrix
    assert np.allclose(result, np.array([[9, 12, 15], [66, 81, 96], [95, 112, 129]])), f"Result is not correct, it is {result} instead of [[9, 12, 15], [66, 81, 96], [95, 114, 133]]"

def test_band_matrix_qr_decomposition():
    band_matrix = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    q, r = band_matrix.qr_decomposition()
    assert np.allclose(q, np.array([[-0.24253563, 0.0878726, -0.96615469], 
                                    [-0.9701425, -0.02196815, 0.24153867], 
                                    [-0, 0.99588946, 0.090577]])), f"Q is not correct, it is {q} instead of [[-0.24253563, 0.0878726, -0.96615469], [-0.9701425, -0.02196815, 0.24153867], [-0, 0.99588946, 0.090577]]"
    assert np.allclose(r, np.array([[-4.12310563, -5.33578375, -5.820855], 
                                    [ 0, 8.03302009, 8.8311962], 
                                    [ 0, 0, 2.26442504]])), f"R is not correct, it is {r} instead of [[4.12, 5.34, 5.82], [0, 8.03, 8.83], [0, 0, 2.26]]"

def test_band_matrix_solve_slae():
    band_matrix = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    b = np.array([1, 2, 3])
    x = band_matrix.solve_slae(b)
    assert np.allclose(x, np.array([0.04, 0.48, -0.09333333])), f"x is not correct, it is {x} instead of [0.04, 0.48, -0.09333333]"

def test_band_matrix_det():
    band_matrix = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    det = band_matrix.det()
    assert np.allclose(det, -75), f"det is not correct, it is {det} instead of -75"

def test_band_matrix_inverse():
    band_matrix = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    inv = band_matrix.inverse()
    assert np.allclose(inv, np.array([[ 0.04, 0.24, -0.16], [ 0.48, -0.12, 0.08], [-0.42666667, 0.10666667, 0.04]])), f"inv is not correct, it is {inv} instead of [[ 0.04, 0.24, -0.16], [ 0.48, -0.12, 0.08], [-0.42666667, 0.10666667, 0.04]]"

def test_band_matrix_zeros():
    band_matrix = BandMatrix.zeros(3, 1, 1)
    assert np.allclose(band_matrix.to_dense(), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])), f"Data is not correct, it is {band_matrix.to_dense()} instead of [[0, 0, 0], [0, 0, 0], [0, 0, 0]]"

def test_band_matrix_empty_like():
    band_matrix = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    empty = band_matrix.empty_like(3, 3)
    assert np.allclose(empty.to_dense(), np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]])), f"Empty is not correct, it is {empty.to_dense()} instead of [[1, 2, 0], [4, 5, 6], [0, 8, 9]]"

def test_band_matrix_to_dense():
    band_matrix = BandMatrix(np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]), 1, 1)
    dense = band_matrix.to_dense()
    assert np.allclose(dense, np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]])), f"Dense is not correct, it is {dense} instead of [[1, 2, 0], [4, 5, 6], [0, 8, 9]]"

def run_test():
    test_band_matrix()
    test_band_matrix_addition()
    test_band_matrix_subtraction()
    test_band_matrix_multiplication()
    test_band_matrix_matrix_multiplication()
    test_band_matrix_qr_decomposition()
    test_band_matrix_solve_slae()
    test_band_matrix_det()
    test_band_matrix_inverse()
    test_band_matrix_zeros()
    test_band_matrix_empty_like()
    test_band_matrix_to_dense()


if __name__ == "__main__":
    run_test()
    


















