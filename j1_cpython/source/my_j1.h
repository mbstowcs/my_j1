/**
 * @file my_j1.h
 * @brief Библиотека для высокопроизводительного вычисления функции Бесселя первого порядка.
 *
 * Используется разбиение диапазона аргументов по их модулям на 3 отрезка, на каждом из которых функция
 * вычисляется по-своему.
 * Для |x| < 6:
 * используется 20 членов ряда Тейлора с суммированием по схеме Горнера.
 * Для 6 <= |x| < 17:
 * используется приближение системой полиномов Чебышева 31-й степени. Коэффициенты получены алгоритмом Ремеза.
 * Для вычисления значения многочлена используется алгоритм Кленшоу.
 * Для 17 <= |x|:
 * используется аппроксимация для больших значений с многочленами 7-й степени при sin и cos.
 *
 * Все функции используют векторизацию avx2, cache-friendly размещение памяти, а также параллелизм с использованием OpenMP.
 */

#pragma once

#include <math.h>
#include <immintrin.h>

#include "j1_poly.c"
#include "j1_large.c"
#include "j1_taylor.c"
#include "avx2_math_stuff.h"

/**
 * Воспомогательная функция для вычисления смещений при распределении значений массива аргументов по их диапазону.
 * @param arr Входной массив
 * @param size Размер массива
 * @param mid_start В этой переменной будет лежать количетсво элементов в диапазоне [left, right)
 * @param hi_start В этой переменной будет лежать size - (количество элементов, >= right)
 */
static inline void calculate_offsets (double *arr, size_t size, size_t *mid_start, size_t *hi_start)
{
    *mid_start = 0, *hi_start = size;

    size_t avx_part_size = size - (size % 4); // вычисляем какую часть массива мы можем обработать инструкциями avx2
    __m256d mm_left = _mm256_set1_pd(left);
    __m256d mm_right = _mm256_set1_pd(right);

    unsigned int lo_4_mask = 0b1111; // маска для инверсии последних битов

    for (size_t i = 0;i < avx_part_size;i += 4)
    {
        __m256d x = avx2_abs_pd(_mm256_loadu_pd(arr + i));
        __m256d cmp_left = _mm256_cmp_pd(x, mm_left, _CMP_LT_OS);
        __m256d cmp_right = _mm256_cmp_pd(x, mm_right, _CMP_LT_OS);

        int mask_left = _mm256_movemask_pd(cmp_left);
        int mask_right = _mm256_movemask_pd(cmp_right);

        *mid_start += __builtin_popcount(mask_left); // считаем количество элементов < left
        *hi_start -= __builtin_popcount(mask_right ^ lo_4_mask); // считаем количество элементов >= right
    }

    // обрабатываем хвост массива
    for (size_t i = avx_part_size;i < size;i++)
    {
        double x = fabs(arr[i]);

        if (x < left)
            (*mid_start)++;
        else if (x >= right)
            (*hi_start)--;
    }
}

/**
 * Комбинированная функция для вычисления j1.
 * @param arr Массив аргументов
 * @param size Размер массива
 * @param result Массив выходных данных
 * @note Если вы знаете, что значения лежат только в каком-то одном диапазоне, лучше вызовите соответствующую функцию.
 * Разбиение массива по отрезкам занимает 60% времени выполнения функции. В будущем это будет оптимизированно.
 */
void j1_combined(double *arr, size_t size, double *result)
{
    double *sorted_arguments = _mm_malloc(size*sizeof(double), 32);
    double *sorted_result = _mm_malloc(size*sizeof(double), 32);

    size_t mid_start, hi_start;
    calculate_offsets(arr, size, &mid_start, &hi_start); // считаем смещения

    // кладем в массив sorted_arguments подряд значения из соответствующих диапазонов
    size_t lo_ind = 0, mid_ind = mid_start, hi_ind = hi_start;

    for (size_t i = 0;i < size;i++)
    {
        double x = fabs(arr[i]);

        if (x < left)
            sorted_arguments[lo_ind++] = arr[i];
        else if (x < right)
            sorted_arguments[mid_ind++] = arr[i];
        else
            sorted_arguments[hi_ind++] = arr[i];
    }

    // обрабатываем каждый диапазон соответствующей функцией
    if (hi_start < size)
        j1_large_values(sorted_arguments + hi_start, size - hi_start, sorted_result + hi_start);
    if (mid_start < hi_start)
        j1_chebyshev(sorted_arguments + mid_start, hi_start - mid_start, sorted_result + mid_start);
    if (mid_start != 0 && hi_start != 0)
        j1_taylor(sorted_arguments, mid_start, sorted_result);

    _mm_free(sorted_arguments);

    lo_ind = 0, mid_ind = mid_start, hi_ind = hi_start;

    for (size_t i = 0;i < size;i++)
    {
        double x = fabs(arr[i]);

        if (x < left)
            result[i] = sorted_result[lo_ind++];
        else if (x < right)
            result[i] = sorted_result[mid_ind++];
        else
            result[i] = sorted_result[hi_ind++];
    }

    _mm_free(sorted_result);
}