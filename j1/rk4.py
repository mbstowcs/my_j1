"""
Вычисление значений функции с помощью метода Рунге-Кутты 4 порядка с динамическим построением сетки по формуле Рунге-Ромберга.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit


# TODO: из-за особенности (v, (1/x**2 - 1)*y - v/x) в нуле вычисление производной таким способом может давать очень неточные результаты. Сделать оценку производной в нуле рядом тейлора.
@njit
def _rk4_j1_F(x: float, y: float, v: float) -> Tuple[float, float]:
    """
    Производная от пары (y, v).
    
    :param x: аргумент
    :param y: значение функции
    :param v: значение производной функции
    :return: пара (производная y, производная v)
    """
    
    if x < 5e-7: # 0 - точка устранимого разрыва, укажем предельные значения
        return 0.5, 0
    return v, (1/x**2 - 1)*y - v/x
    

@njit
def _rk4_step(x: float, y: float, v: float, h: float) -> Tuple[float, float]:
    """
    Один шаг метода Рунге-Кутты 4 порядка.
    
    :param x: начальная точка
    :param y: начальное значение y в точке
    :param v: начальное значение v (производной y) в точке
    :param h: размер шага
    :return: пара (новое y, новое v)
    """
    
    # вычисляем пары коэффициентов
    k1 = _rk4_j1_F(x, y, v)
    k2 = _rk4_j1_F(x + h/2, y + h*k1[0]/2, v + h*k1[1]/2)
    k3 = _rk4_j1_F(x + h/2, y + h*k2[0]/2, v + h*k2[1]/2)
    k4 = _rk4_j1_F(x + h, y + h*k3[0], v + h*k3[1])
    
    return (y + (k1[0] + 2*(k2[0] + k3[0]) + k4[0])*h/6, v + (k1[1] + 2*(k2[1] + k3[1]) + k4[1])*h/6) # возвращаем решение для (y, v)
    

# TODO: сделать оптимальный рассчет eps_v
@njit
def j1_rk4_diap(b: float, h0: float=0.1, eps_y: float=1e-12, eps_v: float=1e-14) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Численное интегрирование уравнения Бесселя 1 порядка j1 используя метод Рунге-Кутта с динамическим построением сетки по формуле Рунге-Ромберга.
    
    :param b: правая граница отрезка
    :param y0: начальное значение y (для 0 можно использовать 1e-6)
    :param v0: начальное значение производной y (для 0 можно использовать 0.5)
    :param h0: начальный шаг
    :param eps_y: требуемая точность y
    :return: пара (сетка аргументов, значения функции на сетке аргументов)
    """
    
    # инициализация начальных значений x, y, v, h
    x = [0]
    y = [0]
    last_v = 0.5
    h = h0
    
    while x[-1] < b: # цикл до правой границы
        eval_h = _rk4_step(x[-1], y[-1], last_v, h) # начальная оценка шага h, двух шагов h и шага 2h
        eval_h2 = _rk4_step(x[-1] + h, *eval_h, h)
        eval_2h = _rk4_step(x[-1], y[-1], last_v, 2*h)
        
        while True: # увеличиваем шаг, если точность не ухудшается
            eval_h_temp = eval_2h
            eval_h2_temp = _rk4_step(x[-1] + 2*h, *eval_h_temp, 2*h)
            eval_2h_temp = _rk4_step(x[-1], y[-1], last_v, 4*h)
            
            if abs(eval_h2_temp[0] - eval_2h_temp[0])/15 < eps_y and abs(eval_h2_temp[1] - eval_2h_temp[1])/15 < eps_v: # если по формуле Рунге-Ромберга точность не ухудшилась
                h *= 2 # увеличиваем шаг
                eval_h, eval_h2, eval_2h = eval_h_temp, eval_h2_temp, eval_2h_temp
            else:
                break
                
        while abs(eval_h2[0] - eval_2h[0])/15 >= eps_y or abs(eval_h2[1] - eval_2h[1])/15 >= eps_v: # уменьшаем шаг, если точность слишком низкая
            h /= 2 # уменьшаем шаг
            eval_2h = eval_h
            eval_h = _rk4_step(x[-1], y[-1], last_v, h)
            eval_h2 = _rk4_step(x[-1] + h, *eval_h, h) # пересчитываем всё для нового шага
                
        x.append(x[-1] + h)
        y.append(eval_h[0])
        last_v = eval_h[1]
        
        
    return np.array(x), np.array(y) # возвращаем посчитанное в виде массивов numpy