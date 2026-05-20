"""
Построение графиков ошибок всех функций по сравнению с эталонной из пакета scipy.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import j1 as j1_scipy

from j1 import j1_taylor, j1_large_values, j1_chebyshev, j1_rk4_diap, j1_integrate_simpson


if __name__ == "__main__":
    # настройка общего вида графика
    
    plt.grid()
    plt.locator_params(axis="x", nbins=20)
    plt.locator_params(axis="y", nbins=10)
    plt.subplots_adjust(left=0.2, right=0.8)
    
    # заголовок и подписи осей
    
    plt.title("Зависимость порядка ошибки по сравнению с эталонной функцией от значения аргумента")
    plt.xlabel(r"x")
    plt.ylabel(r"$\operatorname{log}_{10}|J_1(x) - J^*_1(x)|$")
    
    # параметры отображения (правая граница и количество точек)
    
    right = 150
    count = 30000
    
    # воспомогательные lambda-функции для преобразования координаты в точку в массиве и для рассчета порядка ошибки
    
    cvt = lambda _x: int(_x*count/right)
    err = lambda _x, _y: np.log10(np.abs(_y - j1_scipy(_x)))
    
    # рассчет результатов всех методов
    
    x = np.linspace(0, right, count)
    
    x_row = x[:cvt(15)]
    y_row = j1_taylor(x_row)
    x_large = x[cvt(10):]
    y_large = j1_large_values(x_large)
    x_chebyshev = x[cvt(5):cvt(18)]
    y_chebyshev = j1_chebyshev(x_chebyshev)
    x_rk4, y_rk4 = j1_rk4_diap(40, 0.1)
    x_simpson = x
    y_simpson = j1_integrate_simpson(x)
    
    # вывод графиков с подписями
    
    #plt.plot(x_row, err(x_row, y_row), color=[0.8,0,0], label="Ряд тейлора")
    #plt.plot(x_large, err(x_large, y_large), color=[0,0.8,0], label="Аппроксимация для больших значений")
    #plt.plot(x_chebyshev, err(x_chebyshev, y_chebyshev), color=[0,0,0.8], label="Аппроксимация полиномами Чебышёва")
    #plt.plot(x_rk4, err(x_rk4, y_rk4), color=[0.4,0.4,0], label="Численное решение методом Рунге-Кутты 4 порядка")
    plt.plot(x_simpson, err(x_simpson, y_simpson), color=[0.4,0,0.4], label="Численное решение интеграла методом Симпсона")
    
    #plt.legend()
    plt.show()