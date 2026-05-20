#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("sse3,ssse3,avx2,fma,popcnt")

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#include "my_j1.h"


static PyObject* MYJ1_j1(PyObject* self, PyObject *const *args, Py_ssize_t nargs)
{
    if (__builtin_expect((nargs != 1), false))
    {
        PyErr_SetString(PyExc_TypeError, "j1 takes exactly 1 argument");
        return NULL;
    }

    // конвертируем в массив numpy из любого array-like объекта
    PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF(args[0], NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (__builtin_expect((arr == NULL), false)) // если функция вернула NULL, значит произошла ошибка и код ошибки уже выставлен
        return NULL;

    // получаем размерности массива
    int ndim = PyArray_NDIM(arr);
    npy_intp *dims = PyArray_DIMS(arr);
    npy_intp size = PyArray_SIZE(arr);

    // создаем новый массив для результата с соответствующими размерностями
    PyArrayObject *res = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, NPY_FLOAT64);

    // получаем настоящие буфферы для аргументов и результата
    double *arr_data = (double*)PyArray_DATA(arr);
    double *res_data = (double*)PyArray_DATA(res);

    // применяем функцию j1
    j1_combined(arr_data, size, res_data);

    Py_DECREF(arr); // уменьшаем reference counter для массива, созданного PyArray_FROM_OTF
    return PyArray_Return(res); // возвращаем результат
}

// описания методов
static PyMethodDef my_j1_methods[] = {
        {"j1", (PyCFunction)MYJ1_j1, METH_FASTCALL, "Вычисление j1 для объекта numpy array"},
        {NULL, NULL, 0, NULL}
};

// описание модуля
static struct PyModuleDef my_j1_module = {
        PyModuleDef_HEAD_INIT,
        "my_j1",
        "Реализация numpy-совместимой высокопроизводительной библиотеки для вычисления функции Бесселя 1-го порядка.",
        0,
        my_j1_methods
};

PyMODINIT_FUNC PyInit_my_j1()
{
    import_array() // инициализируем numpy

    if (!__builtin_cpu_supports("avx2"))
    {
        // если процессор не поддерживает avx2, выбрасываем ошибку импорта
        PyErr_SetString(PyExc_ImportError, "avx2 isn't supported by your CPU.");
        return NULL;
    }

    if (PyErr_Occurred()) // если произошла ошибка, то передаем ее
        return NULL;

    return PyModuleDef_Init(&my_j1_module);
}