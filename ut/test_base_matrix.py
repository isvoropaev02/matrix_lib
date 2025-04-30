from matrix_lib import MatrixBase


def run_test():
    mt1 = MatrixBase(3, 4)
    mt2 = MatrixBase.zeros(mt1.m, mt1.n)
    assert mt1.m == mt2.m, f"Num rows do not match: mt1.m = {mt1.m}, mt2.m = {mt2.m}"
    assert mt1.n == mt2.n, f"Num rows do not match: mt1.n = {mt1.n}, mt2.n = {mt2.n}"


if __name__ == "__main__":
    run_test()
