#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from datetime import datetime, timedelta
import time
from math import log, sqrt

# # Лабораторная работа №3
# 
# ### Мотякин Артем Андреевич СКБ211

# 1. Модифицировать (предложить собственные) два метода генерации псевдослучайных чисел.
# 2. Получить не менее 20 выборок каждым методом (диапазон чисел в каждой выборке не менее 5000) объемом не менее 100 элементов каждая.
# 3. Для каждой выборки посчитать среднее, отклонение и коэффициент вариации.
# 4. Каждую выборку проверить на равномерность распределения и случайность, используя критерий Хи-квадрат.
# 5. Для каждого алгоритма осуществить проверку с помощью не менее 3-х тестов NIST и/или Diehard. Сделать выводы и сравнить их с п.4.
# 6. Засечь время генерации чисел от тысячи до миллиона элементов обоими предложенными методами и любым стандартным методом используемого языка программирования. Построить графики сравнения скоростей в зависимости от объема выборки.

# __Конгруэнтный метод пятой степени__: $r_{i+1}=(k_1 * r^5_i + k_2 * r^4_i + k_3 * r^3_i + k_4 * r^2_i + k_5 * r_i + b) \% M$<br>
# Основан на *квадратичном конгруэнтном методе*.

# In[2]:


# параметры программы:
N = 20 #кол-во наборов
n = 160 #кол-во элементов в наборе
path = "./gen_1_results/"

# параметры генератора:
M = 2**31-1
k1 = 23
k2 = 15
k3 = 3
k4 = 78
k5 = 9
r0 = 43
b = 12


# In[3]:


def GenerateAnotherElement(ri):
    return (k1 * ri**5 + k2 * ri**4 + k3 * ri**3 + k4 * ri**2 + k5 * ri**2 + b) % M


# In[4]:


def GenerateSampling(n, r):
    res = []
    a = r
    for i in range(n):
        a = GenerateAnotherElement(a)
        res.append(a)
    return (res, a)


# In[5]:


def GenerateAllSamplings(N, n):
    res = []
    r = r0
    for i in range(N):
        (temp, r) = GenerateSampling(n, r)
        res.append(temp)
        temp_1 = ""
        for j in temp:
            a = round(j / M)
            temp_1 += str(a)
            with open( f"{path}{i}.txt", "w" ) as f:
                f.write(temp_1)
    return res


# In[6]:


def СountMean(sampling):
    return sum(sampling)/len(sampling)

def CountDeviation(sampling, mean_): # sqrt( sum( (x_i - mean)^2 ) / ( n-1 ) )
    s = 0
    for i in sampling:
        s += (i - mean_)**2
    s /= (len(sampling)-1)
    return s ** (0.5)

def CountVariationCoefficient(deviation, mean_): # deviation / mean * 100
    return (deviation / mean_) * 100


# Метод хи-квадрат используется для проверки гипотезы о том, что наблюдаемые данные имеют определенное распределение. Формула для вычисления статистики хи-квадрат выглядит следующим образом:<br>
# $\chi^2 = \sum_{1}^{k} \frac{(O_i-E_i)^2}{E_i}$

# In[7]:


def xi_squere(Sampling, k, n): # хи-квадрат. к - количество отрезков для разбиения, уровень значимости возьмем за 0.95
    delta = (M+1)/k #ширина каждого отрезка
    sampling = sorted(Sampling) #сортируем в порядке возрастания
    tmp = delta #верхняя граница текущего отрезка
    res = 0
    count = 0
    for i in sampling:
        if i < tmp:
            count += 1
        else:
            res += (((count/(n)) - (1/k))**2)/(1/k) #квадрат отклонения доли элементов в отрезке от ожидаемой доли, деленное на ожидаемую долю
            count = 1
            tmp += delta
    res *= n
    return res


# In[8]:


all_samplings = GenerateAllSamplings(N, n)
for sampling in all_samplings:
    mean_ = СountMean(sampling)
    deviation = CountDeviation(sampling, mean_)
    variation_coefficient = CountVariationCoefficient(deviation, mean_)
    print("Среднее: ", mean_)
    print("Отклонение: ", deviation)
    print("Коэффициент вариации: ", variation_coefficient)
    xi_squere_ = xi_squere(sampling, 4, n)
    kvantil = 5.9915 #alpha 0.95
    if xi_squere_ < kvantil:
        print("Принимаем, хи квадрат = ", xi_squere_)
    else:
        print("Не принимаем, хи квадрат = ", xi_squere_)


# In[16]:


get_ipython().system('pip install nistrng')


# In[31]:


import nistrng

all_samplings = GenerateAllSamplings(N, n)
for sampling in all_samplings:

    # Преобразование чисел в биты
    data_bits = np.unpackbits(np.array(sampling).astype(np.uint32).view(np.uint8))

    # Создание экземпляров тестов
    tests = [
        nistrng.SP800_22R1A_BATTERY["monobit"],
        nistrng.SP800_22R1A_BATTERY["frequency_within_block"],
        nistrng.SP800_22R1A_BATTERY["runs"],
        nistrng.SP800_22R1A_BATTERY["longest_run_ones_in_a_block"],
        nistrng.SP800_22R1A_BATTERY["binary_matrix_rank"]
    ]

    eligible_battery: dict = nistrng.check_eligibility_all_battery(data_bits, nistrng.SP800_22R1A_BATTERY)
        
    # Print the eligible tests
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)
        
    # Test the sequence on the eligible tests
    results = nistrng.run_all_battery(data_bits, eligible_battery, False)
    
    # Print results one by one
    print("Test results:")
    for result, elapsed_time in results:
        if result.passed:
            print("- PASSED - score: " + str(np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
        else:
            print("- FAILED - score: " + str(np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")


# __Proposed by George Marsaglia, professor at the University of Florida. Period 2^250, rule is:<br>
# S = 2111111111*x[n-4] + 1492*x[n-3] + 1776*x[n-2] + 5115*x[n-1] + C<br>
# x[n] = S / 2**32__<br>

# Мой алгоритм (по аналогии):<br>
# x[n] = (94604*x[n-5] + 55073*x[n-4] + 3916*x[n-3] + 83045*x[n-2] + 1774*x[n-1] + 76787) % M

# In[33]:


# стартовые параметры программы
X = [86147, 67333, 50210, 43123, 35498] 
path = "./gen_2_results/"


# In[34]:


def GenerateAnotherElement2(x):
    n = len(x)
    S = 94604*x[n-5] + 55073*x[n-4] + 3916*x[n-3] + 83045*x[n-2] + 1774*x[n-1] + 76787
    x.append(S % M)


# In[35]:


def GenerateSampling2(n, x):
    res = x[::]
    for i in range(n):
        GenerateAnotherElement2(res)
    return res


# In[36]:


def GenerateAllSamplings2(N, n):
    res = []
    x = X[::] #для копирования значений массива
    for i in range(N):
        temp = GenerateSampling2(n, x)
        res.append(temp)
        temp1 = ""
        for j in temp:
            a = round(j / M)
            temp1 += str(a)
        with open( f"{path}{i}.txt", "w" ) as f:
            f.write(temp1)
        x = temp[-5:] #5 элементов для генерации 6
    return res


# In[37]:


all_samplings = GenerateAllSamplings2(N, n)
for sampling in all_samplings:
    mean_ = СountMean(sampling)
    deviation = CountDeviation(sampling, mean_)
    variation_coefficient = CountVariationCoefficient(deviation, mean_)
    print("Среднее: ", mean_)
    print("Отклонение: ", deviation)
    print("Коэффициент вариации: ", variation_coefficient)
    xi_squere_ = xi_squere(sampling, 10, n)
    kvantil = 5.9915 #alpha 0.95
    if xi_squere_ < kvantil:
        print("Принимаем, хи квадрат = ", xi_squere_)
    else:
        print("Не принимаем, хи квадрат = ", xi_squere_)
    


# In[38]:


all_samplings = GenerateAllSamplings2(N, n)
for sampling in all_samplings:

    # Преобразование чисел в биты
    data_bits = np.unpackbits(np.array(sampling).astype(np.uint32).view(np.uint8))

    # Создание экземпляров тестов
    tests = [
        nistrng.SP800_22R1A_BATTERY["monobit"],
        nistrng.SP800_22R1A_BATTERY["frequency_within_block"],
        nistrng.SP800_22R1A_BATTERY["runs"],
        nistrng.SP800_22R1A_BATTERY["longest_run_ones_in_a_block"],
        nistrng.SP800_22R1A_BATTERY["binary_matrix_rank"]
    ]

    eligible_battery: dict = nistrng.check_eligibility_all_battery(data_bits, nistrng.SP800_22R1A_BATTERY)
        
    # Print the eligible tests
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)
        
    # Test the sequence on the eligible tests
    results = nistrng.run_all_battery(data_bits, eligible_battery, False)
    
    # Print results one by one
    print("Test results:")
    for result, elapsed_time in results:
        if result.passed:
            print("- PASSED - score: " + str(np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
        else:
            print("- FAILED - score: " + str(np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")




