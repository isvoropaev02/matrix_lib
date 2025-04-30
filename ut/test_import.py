def run_test():
    from matrix_lib import MatrixBase

    _ = MatrixBase.zeros(1, 1)
    # без ассерта, тк исключение выбросит сама ошибка импорта, если произойдет


if __name__ == "__main__":
    run_test()
