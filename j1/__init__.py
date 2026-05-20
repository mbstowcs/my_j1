"""
5 различных методов для оценки функции Бесселя 1-го порядка.

Реализованы оценки через ряд Тейлора, численное интегрирование, численное решение дифференциального уравнения,
алгоритм Ремеза и аппроксимацию для больших значений.
Также, реализована функция, склеенная из 3 частей, дающая точность 10^-15 на всей числовой прямой (пока имеет смысл).
"""

from .integrate import j1_integrate_simpson
from .large import j1_large_values
from .polynomial import j1_chebyshev
from .remez import get_remez_poly
from .rk4 import j1_rk4_diap
from .taylor import j1_taylor, j1_exact_value
from .combined import j1_optimal


__version__ = "0.0.1"

__all__ = [
    "j1_integrate_simpson",
    "j1_large_values",
    "j1_chebyshev",
    "j1_rk4_diap",
    "j1_taylor",
    "j1_exact_value",
    "j1_optimal",
    "get_remez_poly"
]