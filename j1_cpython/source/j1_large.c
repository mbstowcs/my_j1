#pragma once

#include <stdalign.h>
#include <math.h>
#include <immintrin.h>

#include "avx2_math_stuff.h"


#define MIN_PARALLEL_SIZE_LARGE ((size_t)16000) // с какого размера массива можно использовать параллелизм

const int poly_power = 9; // степень многочленов P и Q

// предпосчитанные коэффициенты многочлена P
static const alignas(32) double P_poly_coeffs [] = {450278600.305039, -6656367.71881769, 127641.272646175, -3302.27229448085,
                                                    121.597891876536, -6.88391426810995, 0.676592588424683, -0.144195556640625,
                                                    0.117187500000000, 1.00000000000000};

// предпосчитанные коэффициенты многочлена Q
static const alignas(32) double Q_poly_coeffs [] = {-4043620325.10775, 53104110.1096852, -890297.876707068, 19718.3759122366,
                                                    -603.844076705070, 27.2488273112685, -1.99353173375130, 0.277576446533203,
                                                    -0.102539062500000, 0.375000000000000};

/**
 * @brief Аппроксимация функции бесселя j1 для больших значений.
 * Подобранное количество членов дает точность 1e-15 при числах начиная с 17.
 * @param arr Массив аргументов
 * @param size Размер массива
 * @param result Массив выходных данных
 */
void j1_large_values(double *arr, size_t size, double *result)
{
    __m256d main_coeff = _mm256_set1_pd(sqrt(M_2_PI)); // коэффициент перед асимптотической формулой
    __m256d x_intercept = _mm256_set1_pd(3*M_PI_4); // сдвиг аргумента для тригонометрических функций
    __m256d one = _mm256_set1_pd(1.);

    size_t avx_part_size = size - (size % 4); // вычисляем какую часть массива мы можем обработать инструкциями avx2

    #pragma omp parallel for if(avx_part_size >= MIN_PARALLEL_SIZE_LARGE) \
        shared(avx_part_size, arr, result, x_intercept, one, main_coeff, P_poly_coeffs, Q_poly_coeffs, poly_power) \
        default(none)
    for (size_t i = 0;i < avx_part_size;i+=4)
    {
        __m256d x = _mm256_loadu_pd(arr + i);
        __m256d _x = _mm256_div_pd(one, x); // так как коэффициенты перед sin и cos - многочлены относительно 1/x
        __m256d _x2 = _mm256_mul_pd(_x, _x);
        __m256d _sqrt_x = _mm256_sqrt_pd(_x);

        // вычисляем значения многочлена по схеме Горнера
        __m256d P_poly_value = _mm256_set1_pd(P_poly_coeffs[0]);
        __m256d Q_poly_value = _mm256_set1_pd(Q_poly_coeffs[0]);

        #pragma GCC unroll 9 // размотка цикла для ускорения
        for (int j = 1;j <= poly_power;j++)
        {
            P_poly_value = _mm256_fmadd_pd(P_poly_value, _x2, _mm256_set1_pd(P_poly_coeffs[j]));
            Q_poly_value = _mm256_fmadd_pd(Q_poly_value, _x2, _mm256_set1_pd(Q_poly_coeffs[j]));
        }

        Q_poly_value = _mm256_mul_pd(Q_poly_value, _x); // Q - нечетный многочлен

        x = _mm256_sub_pd(x, x_intercept);
        __m256d cos_part = _mm256_mul_pd(P_poly_value, avx2_cos_pd(x));
        __m256d sin_part = _mm256_mul_pd(Q_poly_value, avx2_sin_pd(x));
        __m256d res = _mm256_mul_pd(main_coeff, _mm256_mul_pd(_sqrt_x, _mm256_sub_pd(cos_part, sin_part)));

        _mm256_storeu_pd(result + i, res);
    }

    // обрабатываем оставшийся хвост массива
    for (size_t i = avx_part_size;i < size;i++)
    {
        double x = arr[i] - 3*M_PI_4;
        double _x = 1/arr[i];
        double _x2 = _x*_x;

        double P_poly_value = P_poly_coeffs[0];
        double Q_poly_value = Q_poly_coeffs[0];

        #pragma GCC unroll 9
        for (int j = 1;j <= poly_power;j++)
        {
            P_poly_value = fma(P_poly_value, _x2, P_poly_coeffs[j]);
            Q_poly_value = fma(Q_poly_value, _x2, Q_poly_coeffs[j]);
        }

        Q_poly_value *= _x;

        result[i] = sqrt(M_2_PI*_x)*(P_poly_value*cos(x) - Q_poly_value*sin(x));
    }
}