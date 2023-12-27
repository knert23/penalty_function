import numpy as np
import math


# Заданная функция
def function(x_1, x_2):
    return np.exp(x_1 ** 2) + (x_1 + x_2) ** 2


# Уравнение плоскости
def plane(x_1, x_2):
    return 0.24 * x_1 - 0.3 * x_2 - 0.3 * function(x_1, x_2) + 1


# Заданная функция со штрафной функцией
def penalty_function(x_1, x_2, penalty_param):
    return function(x_1, x_2) + penalty_param * plane(x_1, x_2) ** 2


# Частная производная штрафной функции по 1 переменной
def first_partial_penalty(x_1, x_2, penalty_param):
    return 2 * x_1 * np.exp(x_1 ** 2) + 2 * x_1 + 2 * x_2 + (-1.2 * x_1 * np.exp(x_1 ** 2) - 1.2 * x_1 - 1.2 * x_2 + 0.48) * (0.24 * x_1 - 0.3 * x_2 - 0.3 * (x_1 + x_2) ** 2 - 0.3 * np.exp(x_1 ** 2) + 1)



# Частная производная штрафной функции по 2 переменной
def second_partial_penalty(x_1, x_2, penalty_param):
    return 2 * x_1 + 2 * x_2 + (-1.2 * x_1 - 1.2 * x_2 - 0.6) * (0.24 * x_1 - 0.3 * x_2 - 0.3 * (x_1 + x_2) ** 2 - 0.3 * np.exp(x_1 ** 2) + 1)


def main():
    e = math.pow(10, -6)
    a = 0.01
    x_0 = [1, 1]
    # Начальное значение штрафного параметра
    penalty_param = 1

    x_min, count = algorithm(e, x_0, a, penalty_param)

    print(f'Минимальное значение функции: {x_min[0], x_min[1]}')
    print(f'Значение функции в данной точке: {penalty_function(x_min[0], x_min[1], penalty_param)}')
    print(f'Количество итераций: {count}')


def algorithm(e, x_0, a, penalty_param):
    count = 0
    max_iterations = 100000
    x_1 = [0, 0]
    while max_iterations > 0:
        x_1[0] = x_0[0] - a * first_partial_penalty(x_0[0], x_0[1], penalty_param)
        x_1[1] = x_0[1] - a * second_partial_penalty(x_0[0], x_0[1], penalty_param)

        count += 1

        if penalty_function(x_1[0], x_1[1], penalty_param) > penalty_function(x_0[0], x_0[1], penalty_param):
            a /= 2

        '''if abs(math.sqrt((x_1[0] - x_0[0]) ** 2 + (x_1[1] - x_0[1]) ** 2)) < e:
            break'''
        if abs(penalty_function(x_1[0], x_1[1], penalty_param) - penalty_function(x_0[0], x_0[1], penalty_param)) < e:
            break
        else:
            # Увеличение штрафного параметра
            penalty_param += 1

        x_0[0] = x_1[0]
        x_0[1] = x_1[1]

        max_iterations -= 1

    return x_1, count


if __name__ == "__main__":
    main()
