"""
Проверка аппроксимации алгоритмом Ремеза.
Выводит оценённую по альтернансу ошибку и реальную ошибку при валидации на тестовой сетке.
"""

import numpy as np
from scipy.special import j1 as j1_scipy

from j1 import get_remez_poly, j1_exact_value, j1_chebyshev


if __name__ == "__main__":
    degree = 25 # максимальная степень многочлена
    a, b = 6.0, 17.0 # левая и правая границы отрезка аппроксимации
    count = 30000 # количество узлов сетки
    test_count = 1000000 # количество узлов валидационной сетки
    
    x_grid = np.linspace(a, b, count)
    y_grid = j1_exact_value(x_grid) # берем точные значения без использования j1 из пакета scipy
    
    coeffs, err = get_remez_poly(degree, x_grid, y_grid, a, b)

    print(f"Коэффициенты: {list(coeffs)}")
    print(f"Степень: {degree}")
    print(f"Оцененная максимальная ошибка: {err:.2e}")

    x_test = np.linspace(a, b, test_count)
    print("Ошибка при валидации: ", np.max(np.abs(j1_scipy(x_test) - j1_chebyshev(x_test, a, b, coeffs))))