def run_test():
    import matrix_lib

    _ = matrix_lib.__name__
    # без ассерта, тк исключение выбросит сама ошибка импорта, если произойдет


if __name__ == "__main__":
    run_test()
