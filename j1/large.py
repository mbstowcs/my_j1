"""
Аппроксимация J1 для больших значений.

Литература:
Abramowitz M., Stegun I. A. Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables. - New York: Dover Publications, 1965. - Section 9.2, Asymptotic Expansions for Bessel Functions.
"""

import numpy as np
from numpy.typing import NDArray
from numba import njit


@njit
def _pn(arr: NDArray[np.float64], depth: int=9) -> NDArray[np.float64]:
    """
    Многочлен при cos в асимптотической формуле.
    
    :param arr: значения аргумента
    :param depth: порядок многочлена, отвечает за точность
    :return: массив значений многочлена
    """
    
    result = np.ones(arr.shape)
    
    for k in range(depth, 0, -1):
        result = 1 - (4 - (4*k - 3)**2)*(4 - (4*k - 1)**2)/(2*k*(2*k - 1)*(8*arr)**2)*result
        
    return result
    
    
@njit
def _qn(arr: NDArray[np.float64], depth: int=9) -> NDArray[np.float64]:
    """
    Многочлен при sin в асимптотической формуле.
    
    :param arr: значения аргумента
    :param depth: порядок многочлена, отвечает за точность
    :return: массив значений многочлена
    """
    
    result = np.ones(arr.shape)
    
    for k in range(depth, 0, -1):
        result = 1 - (4 - (4*k - 1)**2)*(4 - (4*k + 1)**2)/(2*k*(2*k + 1)*(8*arr)**2)*result
        
    result *= 3/(8*arr)
        
    return result

    
@njit
def j1_large_values(arr: NDArray[np.float64], depth: int=9) -> NDArray[np.float64]:
    """
    Асимптотическая формула для больших значений, дает точность 10^-16 начиная с x >= 20 при depth=7
    
    :param arr: значения аргумента
    :param depth: порядок многочлена, отвечает за точность
    :return: массив значений многочлена
    """
    
    pn = _pn(arr, depth)
    qn = _qn(arr, depth)
    
    return np.sqrt(2/(np.pi*arr))*(pn*np.cos(arr - 3*np.pi/4) - qn*np.sin(arr - 3*np.pi/4))