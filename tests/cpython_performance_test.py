import time
import os
from datetime import datetime as dt

import numpy as np
import scipy.special
from matplotlib import pyplot as plt

import my_j1


PER_ELEMENT_TESTS = [(0, 6), (6, 17), (17, 200)]
PER_ELEMENT_N_ITERS = 1000000
ARRAY_TESTS = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
ARRAY_TESTS_SIZE = 5
ARRAY_TESTS_MULTIPLIER = 1000

log_file = None


def print_logs(message):
    global log_file
    
    print(message)
    print(message, file=log_file)


input("Нажмите enter для начала теста производительности...")

directory_name = dt.now().strftime("PerformanceTest_%Y%m%d_%H%M%S")
os.mkdir(directory_name)
os.chdir(directory_name)

log_file = open("per_element_test_log.txt", "w", encoding="utf-8")
print_logs("Запущено поэлементное тестирование")

speedups = []

for diap in PER_ELEMENT_TESTS:
    t1 = time.perf_counter()

    for i in range(PER_ELEMENT_N_ITERS):
        scipy.special.j1(i / PER_ELEMENT_N_ITERS * (diap[1] - diap[0]))
        
    t2 = time.perf_counter()

    for i in range(PER_ELEMENT_N_ITERS):
        my_j1.j1(i / PER_ELEMENT_N_ITERS * (diap[1] - diap[0]))
        
    t3 = time.perf_counter()
    
    scipy_time = t2 - t1
    my_time = t3 - t2
    speedup = scipy_time / my_time
    speedups.append(speedup)

    print_logs(f"Поэлементный тест, {diap[0]} <= x < {diap[1]}, scipy_time = {scipy_time:.6}, my_time = {my_time:.6}, speedup = {speedup:.6}")
    
plt.plot([*range(1, len(speedups) + 1)], speedups)
plt.title("График ускорения в зависимости от номера теста")
plt.xlabel("Номер теста")
plt.ylabel(r"Ускорение $\frac{T_{scipy}}{T_{my}}$")
plt.grid()

plt.savefig("per_element_test.png", dpi=300)
log_file.close()

plt.cla()
    
log_file = open("array_test_log.txt", "w", encoding="utf-8")
print_logs("Запущено тестирование на массивах")

speedups = []

for n in ARRAY_TESTS:
    sum_scipy_time = 0
    sum_my_time = 0
    
    print_logs(f"размер массива = {n}")
    
    for i in range(1, ARRAY_TESTS_SIZE + 1):
        arr = np.random.rand(n) * ARRAY_TESTS_MULTIPLIER
        
        t1 = time.perf_counter()
        scipy_result = scipy.special.j1(arr)
        t2 = time.perf_counter()
        my_result = my_j1.j1(arr)
        t3 = time.perf_counter()
        
        scipy_time = t2 - t1
        my_time = t3 - t2
        speedup = scipy_time / my_time
        
        sum_scipy_time += scipy_time
        sum_my_time += my_time
        
        print_logs(f"тест {i} : scipy_time = {scipy_time:.6}, my_time = {my_time:.6}, speedup = {speedup:.6}")
        
    mean_scipy_time = sum_scipy_time / ARRAY_TESTS_SIZE
    mean_my_time = sum_my_time / ARRAY_TESTS_SIZE
    mean_speedup = mean_scipy_time / mean_my_time
        
    speedups.append(mean_speedup)
       
    print_logs(f"среднее scipy_time: {mean_scipy_time:.6}")
    print_logs(f"среднее my_time: {mean_my_time:.6}")
    print_logs(f"среднее speedup: {mean_speedup:.6}")
    print_logs("-"*20)
    
plt.plot(np.log10(ARRAY_TESTS), speedups)
plt.title("График ускорения в зависимости от порядка размера массива")
plt.xlabel("Порядок размера массива")
plt.ylabel(r"Ускорение $\frac{T_{scipy}}{T_{my}}$")
plt.grid()

plt.savefig("array_test.png", dpi=300)
log_file.close()

print(f"Результаты тестов сохранены в папке {directory_name}")
input("Тест завершен, нажмите enter для выхода...")