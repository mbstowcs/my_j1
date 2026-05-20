#pragma once

#include <stdalign.h>
#include <math.h>
#include <immintrin.h>


#define MIN_PARALLEL_SIZE_TAYLOR ((size_t)25000) // с какого размера массива можно использовать параллелизм

static const int j1_taylor_coeffs_size = 20; // количество коэффициентов

// предпосчитанные коэффициенты ряда Маклорена для j1
static const alignas(32) double j1_taylor_coeffs [] = {-6.146260044082943e-48,9.342315267006073e-45,-1.278028728526431e-41,1.5643071637163513e-38,
                                                       -1.7019661941233902e-35,1.6338875463584545e-32,-1.3724655389411017e-29,9.99154912349122e-27,
                                                       -6.234726653058522e-24,3.2919356728148996e-21,-1.448451696038556e-18,5.214426105738801e-16,
                                                       -1.5017547184527747e-13,3.3639305693342155e-11,-5.651403356481482e-09,6.781684027777778e-07,
                                                       -5.425347222222222e-05,0.0026041666666666665,-0.0625,0.5};

/**
 * @brief Аппроксимация j1 формулой Тейлора.
 * Подобранное количество членов дает точность 1e-15 при |x| до 6.
 * @param arr Массив аргументов
 * @param size Размер массива
 * @param result Массив выходных данных
 */
void j1_taylor(double *arr, size_t size, double *result)
{
    size_t avx_part_size = size - (size % 4); // вычисляем какахаую часть массива мы можем обработать инструкциями avx2

    #pragma omp parallel for if(avx_part_size >= MIN_PARALLEL_SIZE_TAYLOR) \
        shared(avx_part_size, arr, j1_taylor_coeffs, j1_taylor_coeffs_size, result) \
        default(none)
    for (size_t i = 0;i < avx_part_size;i+=4)
    {
        __m256d x = _mm256_loadu_pd(arr + i);
        __m256d x2 = _mm256_mul_pd(x, x);

        // суммируем многочлен Тейлора по схеме Горнера
        __m256d res = _mm256_set1_pd(j1_taylor_coeffs[0]);

        #pragma GCC unroll 19 // размотка цикла для ускорения
        for (int j = 1;j < j1_taylor_coeffs_size;j++)
            res = _mm256_fmadd_pd(res, x2, _mm256_set1_pd(j1_taylor_coeffs[j]));

        res = _mm256_mul_pd(res, x);

        _mm256_storeu_pd(result + i, res);
    }

    // обрабатываем оставшийся хвост массива
    for (size_t i = avx_part_size;i < size;i++)
    {
        double x = arr[i];
        double x2 = x*x;

        double res = j1_taylor_coeffs[0];

        #pragma GCC unroll 19
        for (int j = 1;j < j1_taylor_coeffs_size;j++)
            res = fma(res, x2, j1_taylor_coeffs[j]);

        res *= x;

        result[i] = res;
    }
}