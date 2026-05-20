/**
 * @file avx2_math_stuff.h
 * @brief Высокопроизводительная реализация функций sin и cos для avx2 в режиме 4 double (pd).
 *
 * Функции суммируют 11 членов разложения sin и cos в ряд Маклорена по схеме Горнера.
 * Для приведения аргумента в диапазон [-pi/2, pi/2] используется редукция Коди-Уэйта.
 */

#pragma once

#include <stdalign.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>

const int cos_poly_power = 10; // степень полинома косинуса
const int sin_poly_power = 10; // степень полинома синуса

// предпосчитанные коэффициенты для косинуса (знакочередующиеся обратные факториалы четных чисел)
static const alignas(32) double cos_coeffs [] = {4.110317623312165e-19,-1.5619206968586225e-16,4.779477332387385e-14,-1.1470745597729725e-11,
                                                 2.08767569878681e-09,-2.755731922398589e-07,2.48015873015873e-05,-0.001388888888888889,
                                                 0.041666666666666664,-0.5,1};

// предпосчитанные коэффициенты для синуса (знакочередующиеся обратные факториалы нечетных чисел)
static const alignas(32) double sin_coeffs [] = {1.9572941063391263e-20,-8.22063524662433e-18,2.8114572543455206e-15,-7.647163731819816e-13,
                                                 1.6059043836821613e-10,-2.505210838544172e-08,2.7557319223985893e-06,-0.0001984126984126984,
                                                 0.008333333333333333,-0.16666666666666666,1};

/**
 * @brief Реализация модуля для avx2-вектора из 4 double
 * @param x Аргументы модуля
 * @return Возвращает avx2-вектор значений модуля
 */
__attribute__((__always_inline__))
static inline __m256d avx2_abs_pd(__m256d x)
{
    __m256d sign = _mm256_set1_pd(-0.0);

    // сбрасываем бит знака
    return  _mm256_andnot_pd(sign, x);
}

/**
 * @brief Преобразует вектор из 4 положительных double в вектор из 4 ulong64
 * @param x Входной вектор из 4 double
 * @return Вектор из 4 ulong64
 * @warning Метод работает, если число меньше, чем 2^52
 */
__attribute__((__always_inline__))
static inline __m256i avx2_convert_positive_f64_to_ull(__m256d x)
{
    // мантисса double имеет 52 бита, так что сложение с 2^52 записанным только
    // в мантиссу заставит хранить число только в мантиссе, занулив экспоненту

    __m256d cvt_magic = _mm256_set1_pd(0x0010000000000000);
    x = _mm256_add_pd(x, cvt_magic);

    // убираем старший бит мантиссы, который мы прибавили
    return _mm256_xor_si256(_mm256_castpd_si256(x), _mm256_castpd_si256(cvt_magic));
}

/**
 * @brief Реализует редукцию Коди-Уэйта для приведения avx2-вектора аргументов в диапазон [-pi/2, pi/2]
 * @param x Вектор аргументов
 * @param sign_mask Вектор-маска нечетности сдвигов
 * @return Приведенный вектор аргументов
 */
__attribute__((__always_inline__))
static inline __m256d avx2_cody_waite_reduction_pd(__m256d x, __m256i *sign_mask)
{
    // редукция Коди-Уэйта
    // используем точное представление числа пи в 16-ричном виде

    __m256d pi_hi = _mm256_set1_pd(0x3.243F6A8885A3p+0);
    __m256d pi_lo = _mm256_set1_pd(0x0.8D313198A2E0p-52);

    __m256d period_2pi = _mm256_round_pd(_mm256_div_pd(x, pi_hi), _MM_FROUND_TO_NEAREST_INT);
    x = _mm256_fnmadd_pd(period_2pi, pi_hi, x); // вычитаем старшие биты
    x = _mm256_fnmadd_pd(period_2pi, pi_lo, x); // вычитаем младшие биты
    *sign_mask = _mm256_and_si256(avx2_convert_positive_f64_to_ull(period_2pi), _mm256_set1_epi64x(1));
    *sign_mask = _mm256_slli_epi64(*sign_mask, 63); // поскольку мы вычли pi*n, в зависимости от четности n знак может
    // оказаться отрицательным, так что мы сохраняем маску четности для дальнейшего восстановления знака

    return x;
}

/**
 * @brief Реализация косинуса для avx2-вектора из 4 double.
 * Метод использует редукцию Коди-Уэйта до диапазона [-pi/2, pi/2] и 11 членов ряда Маклорена разложения косинуса.
 * @param x Вектор аргументов
 * @return Вектор значений косинуса
 */
__attribute__((__always_inline__))
static inline __m256d avx2_cos_pd(__m256d x)
{
    x = avx2_abs_pd(x); // используем четность косинуса

    // редукция Коди-Уэйта
    __m256i sign_mask;
    x = avx2_cody_waite_reduction_pd(x, &sign_mask);

    // суммируем 11 членов ряда Маклорена по схеме Горнера
    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d result = _mm256_set1_pd(cos_coeffs[0]);

    #pragma GCC unroll 10 // размотка цикла для ускорения
    for (int i = 1;i <= cos_poly_power;i++)
        result = _mm256_fmadd_pd(result, x2, _mm256_set1_pd(cos_coeffs[i]));

    // восстанавливаем сохраненный знак числа
    result = _mm256_xor_pd(result, _mm256_castsi256_pd(sign_mask));

    return result;
}

/**
 * @brief Реализация синуса для avx2-вектора из 4 double.
 * Метод использует редукцию Коди-Уэйта до диапазона [-pi/2, pi/2] и 11 членов ряда Маклорена разложения синуса.
 * @param x Вектор аргументов
 * @return Вектор значений синуса
 */
__attribute__((__always_inline__))
static inline __m256d avx2_sin_pd(__m256d x)
{
    __m256i sign_mask = _mm256_castpd_si256(_mm256_and_pd(x, _mm256_set1_pd(-0.0)));
    x = avx2_abs_pd(x); // используем нечетность синуса

    // редукция Коди-Уэйта
    __m256i reduction_sign_mask;
    x = avx2_cody_waite_reduction_pd(x, &reduction_sign_mask);
    sign_mask = _mm256_xor_si256(sign_mask, reduction_sign_mask);

    // суммируем 11 членов ряда Маклорена по схеме Горнера
    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d result = _mm256_set1_pd(sin_coeffs[0]);

    #pragma GCC unroll 10 // размотка цикла для ускорения
    for (int i = 1;i <= sin_poly_power;i++)
        result = _mm256_fmadd_pd(result, x2, _mm256_set1_pd(sin_coeffs[i]));

    result = _mm256_mul_pd(result, x);

    // восстанавливаем сохраненный знак числа
    result = _mm256_xor_pd(result, _mm256_castsi256_pd(sign_mask));

    return result;
}