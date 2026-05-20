"""
Аппроксимации для J1, использующие разложение в ряд Тейлора.
"""

from decimal import Decimal as D
from decimal import getcontext

import numpy as np
from numpy.typing import NDArray
from numba import njit


# TODO: оценка остаточного члена в форме Лагранжа, суммирование с конца
@njit
def j1_taylor(arr: NDArray[np.float64], max_iters: int=50, eps: float=5e-16) -> NDArray[np.float64]:
    """
    Разложение функции Бесселя в ряд тейлора.
    Дает точность 10^-16 при значениях от 0 до 5-6.
    
    :param arr: значения аргумента
    :param max_iters: максимальный член суммы (по умолчанию 50)
    :param eps: требуемая точность
    :return: массив значений функции
    """
    
    result = np.empty_like(arr)

    for j in range(arr.shape[0]):
        x = arr[j]

        term = x / 2
        series = term

        for k in range(1, max_iters + 1):
            term *= -x * x / (4.0 * k * (k + 1.0))
            series += term

            if abs(term) < eps * abs(series): # относительное условие прекращения
                break

        result[j] = series

    return result
    
    
def j1_exact_value(arr: NDArray[np.float64], eps: float=5e-16) -> NDArray[np.float64]:
    """
    Значение с точностью до 10^-16 через ряд Тейлора с длинной арифметикой, работает как минимум до 20.
    Неприменим для быстрых рассчетов.
    
    :param arr: значения аргумента
    :param eps: требуемая точность
    :return: массив значений функции
    """
    
    old_prec = getcontext().prec # сохраняем старую точность
    getcontext().prec = 100 # устанавливаем точность 100 знаков для Decimal
    
    try:
        result = np.empty_like(arr)
        eps = D(eps)
    
        for i in range(len(arr)):
            x = D(arr[i])
            term = x / 2
            s = term

            for k in range(1, 10000):
                term *= -x * x / (4 * k * (k + 1))
                s += term
                
                if abs(term) < eps*abs(s):
                    break
                    
            result[i] = float(s)

        return result
        
    finally:
        getcontext().prec = old_prec # восстанавливаем старую точность