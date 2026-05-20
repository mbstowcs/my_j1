"""
Комбинированная реализация из трёх функций, хорошо работающих на своих диапазонах.
"""

import numpy as np
from numpy.typing import NDArray
from numba import njit

from .taylor import j1_taylor
from .polynomial import j1_chebyshev
from .large import j1_large_values


@njit
def j1_optimal(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Функция, скомбинированная из трёх аппроксимаций, дающая хорошую точность при любых значениях аргумента.
    
    :param arr: значения аргумента
    :return: массив значений функции
    """
    
    result = np.empty_like(arr)
    abs_arr = np.abs(arr)
    
    small_ind = abs_arr < 6 # если x лежит в [0, 6), то используем ряд Тейлора
    medium_ind = np.logical_and(6 <= abs_arr, abs_arr < 17) # если x лежит в [6, 20), то используем полиномиальную аппроксимацию
    large_ind = 17 <= abs_arr # если x лежит в [20, +inf), то используем аппроксимацию для больших значений
    
    result[small_ind] = j1_taylor(arr[small_ind])
    result[medium_ind] = j1_chebyshev(arr[medium_ind])
    result[large_ind] = j1_large_values(arr[large_ind])
    
    result[arr < 0] = -result[arr < 0] # так как функция нечётная
    
    return result