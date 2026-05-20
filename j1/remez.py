"""
Аппроксимация функции на отрезке полиномами Чебышёва с использованием алгоритма Ремеза с эффективным множественным обменом и масштабированием.

Литература:
Cheney E. W. Introduction to Approximation Theory. - New York: McGraw-Hill, 1966.
Karam, L., & McClellan, J. H. (1994). Multiple exchange Remez algorithm for complex FIR filter design in the Chebyshev sense. In Proceedings - IEEE International Symposium on Circuits and Systems (Vol. 2, pp. 517-520). IEEE. 
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit
from numpy.polynomial.chebyshev import chebvander, chebval


@njit
def _chebyshev_nodes(n: int, a: float, b: float) -> NDArray[np.float64]:
    """
    Вычисление узлов Чебышёва на отрезке от a до b.
    
    :param n: количество узлов
    :param a: левая граница отрезка
    :param b: правая граница отрезка
    :return: массив, содержащий узлы Чебышёва
    """
    
    k = np.arange(n)
    x = np.cos((2*k + 1) * np.pi / (2*n))
    
    return (a + b)/2 + (b - a)*x/2
    
    
@njit
def _to_scaled(arr: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """
    Выполняет масштабирование значений на отрезок [-1, 1].
    
    :param arr: значения аргумента
    :param a: левая граница исходного отрезка
    :param b: правая граница исходного отрезка
    :return: массив, содержащий масштабированные значения
    """
    
    return 2 * (arr - a) / (b - a) - 1
    

def get_remez_poly(n: int, x_grid: NDArray[np.float64], y_grid: NDArray[np.float64], a: float, b: float, max_iters: int=25, eps: float=5e-16) -> Tuple[NDArray[np.float64], float]:
    """
    Строит приближение функции полиномами Чебышёва на заданном отрезке по сетке, используя алгоритм Ремеза с множественным обменом.
    
    :param n: степень многочлена
    :param x_grid: значения аргумента
    :param y_grid: значение функции на сетке аргументов
    :param a: левая граница отрезка аппроксимации
    :param b: правая граница отрезка аппроксимации
    :param max_iters: максимальное количество итераций алгоритма
    :param eps: требуемая точность
    :return: пара (коэффициенты аппроксимации в обратном порядке, ошибка аппроксимации)
    """
    
    x_scaled = _to_scaled(x_grid, a, b) # масштабируем x до [-1, 1]
    nodes_scaled = _chebyshev_nodes(n + 2, -1, 1) 
    nodes_idx = np.searchsorted(x_scaled, nodes_scaled)
    
    best_poly_coeffs = None
    min_max_error = np.inf

    for _ in range(max_iters):
        curr_nodes_x = x_scaled[nodes_idx]
        curr_nodes_y = y_grid[nodes_idx]

        vander = chebvander(curr_nodes_x, n) # строим альтернативную матрицу для полиномов Чебышёва
        dm = np.ones(n + 2)
        dm[1::2] = -1 # чередование 1, -1, 1, -1, ... по алгоритму Ремеза
        A = np.hstack([vander, dm[:, None]])
        
        try:
            res = np.linalg.solve(A, curr_nodes_y) # пытаемся решить линейное уравнение
            
        except np.linalg.LinAlgError:
            break # если оно выбросило ошибку, значит, скорее всего, матрица системы выродилась
            
        coeffs = res[:-1] # построенные коэффициенты
        E = res[-1]
        
        y_pred = chebval(x_scaled, coeffs)
        error = y_grid - y_pred # вычисляем ошибку предсказанного значения
        abs_err = np.abs(error)
        max_err = np.max(abs_err)
        
        if max_err < min_max_error: # если текущее приближение лучше, выбираем его
            min_max_error = max_err
            best_poly_coeffs = coeffs

        diff = np.diff(np.sign(np.diff(error))) # np.diff считает численную производную ошибки, np.sign считает знаки, а дальнейший np.diff дает нули, если производная не меняется и 2 (-2) если меняется. Так мы можем определить все локальные экстремумы
        extrema_idx = np.where(diff != 0)[0] + 1 # определяем индексы где производная не равна 0
        extrema_idx = np.unique(np.concatenate([[0], extrema_idx, [len(x_grid)-1]])) # добавляем границы отрезка, если их еще там нет
        
        filtered_extrema = [] # в общем случае набор экстремумов - подряд идущие группы отрицательных и положительных значений. Можно выбрать из каждой такой группы подряд идущих значений максимальный элемент.
        ind = 0
        for i in range(len(extrema_idx)):
            if i == len(extrema_idx) - 1 or error[extrema_idx[i]]*error[extrema_idx[i+1]] < 0: # если элемент последний или знак поменялся
                group_indices = extrema_idx[ind:i + 1]
                best_in_group = group_indices[np.argmax(abs_err[group_indices])]
                filtered_extrema.append(best_in_group) # выбираем максимальный по модулю из группы и добавляем в массив
                ind = i + 1

        filtered_extrema = np.array(filtered_extrema) 
        
        """
        Эвристика для множественного обмена заключается в том, что мы будем выбирать такой кандидат на альтернанс,
        что сумма модулей ошибки на всех значениях будет максимальна.
        Мы будем это делать с помощью динамики по путям.
        """

        dp = np.full((len(filtered_extrema), n + 2), -1.0) # dp[i][j] - максимальная сумма для префикса длины j + 1 начиная с i-й позиции.
        parent = np.full((len(filtered_extrema), n + 2), -1, dtype=int) # сохраняем откуда префикс для восстановления путей

        max_reachable = np.full((n + 2, 2), -1.0) # max_reachable[j][sgn] - посчитанное максимально достижимое значение суммы на префиксе длиной j + 1
        max_idx_tracker = np.full((n + 2, 2), -1, dtype=int) # сохраняем откуда префикс для восстановления ответа

        for i in range(len(filtered_extrema)):
            val = abs_err[filtered_extrema[i]]
            sgn = int(error[filtered_extrema[i]] > 0)
            
            dp[i, 0] = val # база динамики - путь длины 1
            
            for j in range(1, n + 2): # переход динамики
                prev_best_sum = max_reachable[j - 1, 1 - sgn]
                if prev_best_sum >= 0:
                    dp[i, j] = val + prev_best_sum
                    parent[i, j] = max_idx_tracker[j - 1, 1 - sgn]
                    
            for j in range(n + 2): # обновляем максимумы
                if dp[i, j] > max_reachable[j, sgn]:
                    max_reachable[j, sgn] = dp[i, j]
                    max_idx_tracker[j, sgn] = i

        last_idx = np.argmax(dp[:, n + 2 - 1])
        final_nodes = []
        for j in range(n + 1, -1, -1): # проходим путь в обратном порядке и восстанавливаем ответ
            final_nodes.append(filtered_extrema[last_idx])
            last_idx = parent[last_idx, j]

        nodes_idx = np.array(final_nodes[::-1])
            
        if abs(max_err - abs(E)) / max_err < eps: # если вычисленное E больше всего похоже на максимальную ошибку, то по теореме Чебышёва об альтернансе, мы нашли наиболее похожее приближение в заданной точности
            break

    return best_poly_coeffs[::-1], min_max_error # возвращаем коэффициенты в обратном порядке для удобства использования в алгоритме Кленшоу
    