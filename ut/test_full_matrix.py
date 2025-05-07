import numpy as np

from matrix_lib import FullMatrix  # импортировать именно из matrixlib

height_and_width = [
    (1, 5),
    (10, 1),
    (6, 13),
    (4, 4),
]  # перебрать несколько крайних случаев


def run_test():
    for shape in height_and_width:
        mt1 = FullMatrix.zero(shape[0], shape[1], 0)
        mt2 = FullMatrix.zero(shape[0], shape[1], 3.14)
        assert (mt1.dtype == np.int64) and (
            mt2.dtype == np.float64
        ), f"dtype does not change: mt1.dtype = {mt1.dtype}, mt2.m = {mt2.dtype}"
        assert (
            mt1.height == mt2.height
        ), f"Num rows do not match: mt1.height = {mt1.height}, mt2.height = {mt2.height}"
        assert (
            mt1.width == mt2.width
        ), f"Num cols do not match: mt1.width = {mt1.width}, mt2.width = {mt2.width}"
        mt3 = FullMatrix.empty_like(mt2)
        mt4 = FullMatrix.empty_like(mt1, width=1, height=1)
        assert (
            mt2.shape == mt3.shape
        ), f"shape mismatch after empty_like: mt2.shape = {mt1.shape}, mt3.shape = {mt2.shape}"
        assert (
            mt4.shape != mt1.shape
        ), f"shape did not change after empty_like with shape cast: mt1.shape = {mt1.shape}, mt4.shape = {mt4.shape}"
        assert (
            mt4.dtype == mt1.dtype
        ), f"dtype change after empty_like: mt1.dtype = {mt1.dtype}, mt4.dtype = {mt4.dtype}"
        assert (
            mt3.dtype == mt2.dtype
        ), f"dtype change after empty_like with shape cast: mt2.dtype = {mt2.dtype}, mt3.dtype = {mt3.dtype}"


if __name__ == "__main__":
    run_test()
