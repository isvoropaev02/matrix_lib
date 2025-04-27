from matrix_lib import MatrixBase

mt1 = MatrixBase(3, 4)
print(mt1._mat)  # ради ut можно и сделать так
print(mt1.m)
print(mt1.n)

mt2 = MatrixBase.zeros(3, 4)
print(mt2._mat)
print(mt2.m)
print(mt2.n)

# далее тест будет расширен и добавлены assert-проверки
