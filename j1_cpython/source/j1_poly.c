#pragma once

#include <math.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdbool.h>


#define MIN_PARALLEL_SIZE_POLY ((size_t)11000) // с какого размера массива можно использовать параллелизм
const int remez_coeffs_size = 26; // количество коэффициентов

// предпосчитанные коэффициенты аппроксимации полиномом Чебышёва для отрезка [6, 20]
static const alignas(32) double remez_poly_coeffs [] = {4.32159739601054e-16, -1.2231546647260535e-14, -3.134946967480416e-14, 9.018069380260976e-13,
                                                        1.9832881179789817e-12, -5.572446492892421e-11, -1.0219294012734787e-10, 2.8210484577224778e-09,
                                                        4.162787586643671e-09, -1.1429612724948976e-07, -1.2874015011818147e-07, 3.595725647181294e-06,
                                                        2.8381392663118567e-06, -8.435335073339732e-05, -3.958083854932352e-05, 0.0013933509026637983,
                                                        0.0002422925836196856, -0.0148306096983199, 0.0012328041762905274, 0.08680989717984268,
                                                        -0.023872216062040624, -0.1891141971794695, 0.06236950971627398, -0.06760562401877901,
                                                        0.049572159647635006, -0.0037481234736116867};

const double left = 6.; // левая граница
const double right = 17.; // правая граница

const double intercept = 2 / (right - left); // intercept и slope для быстрого масштабирования аргумента
const double slope = - 2*left / (right - left) - 1; // одной инструкцией fmadd

/**
 * @brief Высокопроизводительная оценка функции j1 на отрезке [a, b],
 * аппроксимированной полиномами Чебышёва с помощью алгоритма Кленшоу.
 * @param arr Массив аргументов
 * @param size Размер массива
 * @param result Массив выходных данных
 */
void j1_chebyshev(double *arr, size_t size, double *result)
{
    __m256d sign = _mm256_set1_pd(-0.0);
    __m256d mm_intercept = _mm256_set1_pd(intercept);
    __m256d mm_slope = _mm256_set1_pd(slope); // коэффициенты приведения x к отрезку [-1,1]

    size_t avx_part_size = size - (size % 4); // вычисляем какую часть массива мы можем обработать инструкциями avx2

    // распараллеливаем цикл если размер обрабатываемой части >= MIN_PARALLEL_SIZE
    #pragma omp parallel for if(avx_part_size >= MIN_PARALLEL_SIZE_POLY) \
        shared(avx_part_size, arr, result, mm_intercept, mm_slope, remez_poly_coeffs, remez_coeffs_size, sign) \
        default(none)
    for (size_t i = 0;i < avx_part_size;i+=4)
    {
        __m256d element = _mm256_loadu_pd(arr + i); // загружаем аргументы
        __m256d sign_mask = _mm256_and_pd(element, sign); // сохраняем маску знака
        element = _mm256_andnot_pd(sign, element); // сбрасываем бит знака
        element = _mm256_fmadd_pd(mm_intercept, element, mm_slope); // масштабируем аргументы инструкцией fmadd

        // инициализация b1, b2 для алгоритма Кленшоу
        __m256d double_element = _mm256_add_pd(element, element);
        __m256d b1 = _mm256_setzero_pd(), b2 = _mm256_setzero_pd();
        __m256d b3; // воспомогательная переменная

        // вычисляем переходы алгоритма как b2, b1 = b1, coeff + 2*element*b1 - b2

        #pragma GCC unroll 26 // размотка цикла для ускорения
        for (size_t j = 0;j < remez_coeffs_size;j++)
        {
            __m256d coeff = _mm256_set1_pd(remez_poly_coeffs[j]);

            b3 = b2;
            b2 = b1;
            b1 = _mm256_add_pd(coeff, _mm256_fmsub_pd(b1, double_element, b3));
        }

        __m256d res = _mm256_fnmadd_pd(b2, element, b1); // сохраняем b1 - element*b2 в качестве результата
        res = _mm256_xor_pd(res, sign_mask); // восстанавливаем знак
        _mm256_storeu_pd(result + i, res);
    }

    // обрабатываем оставшийся хвост массива
    for (size_t i = avx_part_size;i < size;i++)
    {
        double b1 = 0, b2 = 0, b3;

        double x = arr[i];
        bool sign_flag = x < 0 ? true : false;
        x = fabs(x);
        double element = intercept*x + slope;

        #pragma GCC unroll 26 // размотка цикла для ускорения
        for (size_t j = 0;j < remez_coeffs_size;j++)
        {
            b3 = b2;
            b2 = b1;
            b1 = fma(2*element, b1, remez_poly_coeffs[j] - b3);
        }

        double res = fma(-element, b2, b1);
        res = sign_flag ? -res : res;

        result[i] = res;
    }
}