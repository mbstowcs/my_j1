"""
Вычисление значений функции с помощью численного интегрирования методом Симпсона.
"""

import numpy as np
from numpy.typing import NDArray
from numba import njit


@njit
def j1_integrate_simpson(arr: NDArray[np.float64], max_iters: int=24, eps: float=5e-16) -> NDArray[np.float64]:
    """
    Вычисление j1 как интеграла по методу Симпсона.
    В функции используется рекурсивный пересчёт метода Симпсона по формуле: S_{new} = S_{old}/2 - S_{old\ odd}/6 + S_{new\ nodes}*h*2/3
    
    :param arr: массив значений аргумента
    :param max_iters: максимальное количество удвоений количества отрезков подразбиения
    :param eps: требуемая точность
    :return: массив значений функции
    """
    
    result = np.empty_like(arr)
    
    for x in range(len(arr)):
        h = np.pi
        last = float("inf") # начальное большое значение
        sum_even = 0.
        sum_odd = np.sin(arr[x])*h*2/(3*np.pi)
        new = sum_odd
        
        for i in range(max_iters): # максимальное количество итераций (2**24 это примерно 2*1e7, будет работать медленно для большого количества x)
            if np.abs(new - last)/15 < eps: # формула Рунге-Ромберга
                break
             
            h /= 2. # уменьшаем шаг вдвое
            new_nodes = np.arange(h/2., np.pi, h) # вычисляем значения на новых узлах
            
            sum_even = sum_even/2. + sum_odd/4. # пересчитываем сумму на чётных узлах
            sum_odd = np.sum(np.cos(new_nodes - arr[x]*np.sin(new_nodes)))*h*2/(3*np.pi) # считаем новую сумму на нечётных узлах
            
            last, new = new, sum_even + sum_odd # обновляем последнее и предпоследнее значения

        result[x] = new
        
    return result