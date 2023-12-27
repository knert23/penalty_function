import numpy as np
import math


# Заданная функция
def function(x):
    return np.exp(x[0] ** 2) + (x[0] + x[1]) ** 2
    #return math.sqrt(x[0] ** 2 + x[1] ** 2 + 1) + x[0] / 2 - x[1] / 2


# Уравнение плоскости
def plane(x):
    return 0.24 * x[0] - 0.3 * x[1] - 0.3 * function(x) + 1
    #return x[0] - x[1] + function(x) - 1


# Заданная функция со штрафной функцией
def penalty_function(x, penalty_param):
    return function(x) + penalty_param * plane(x) ** 2


def nelder_mead(f, initial_point, penalty_param, e=1e-6, max_iter=1000, alpha=1, beta=0.5, gamma=2):
    reflection_coef = alpha
    contraction_coef = beta
    expansion_coef = gamma

    # Генерация начального многоугольника
    polyhedron = generate_initial_polyhedron(initial_point)
    values = np.array([f(point, penalty_param) for point in polyhedron])

    count = 0
    for k in range(max_iter):
        # Сортировка
        order = np.argsort(values)
        best, worst, second_worst = order[0], order[-1], order[-2]
        # Нахождение центра масс
        centroid = np.mean([polyhedron[i] for i in order[:-1]], axis=0)

        # Отражение
        x_r = (1 + reflection_coef) * centroid - reflection_coef * polyhedron[worst]
        f_r = f(x_r, penalty_param)

        # Растяжение
        if f_r < values[best]:
            x_e = (1 - expansion_coef) * centroid + expansion_coef * x_r
            f_e = f(x_e, penalty_param)
            if f_e < f_r:
                polyhedron[worst] = x_e
                values[worst] = f_e
            else:
                polyhedron[worst] = x_r
                values[worst] = f_r
        # Сжатие
        else:
            x_k = (1 - contraction_coef) * centroid + contraction_coef * x_r
            f_k = f(x_k, penalty_param)
            if f_k < values[worst]:
                polyhedron[worst] = x_k
                values[worst] = f_k
            # Редукция
            else:
                for i in order[1:]:
                    polyhedron[i] = 0.5 * (polyhedron[i] + polyhedron[best])
                    values[i] = f(polyhedron[i], penalty_param)

        count += 1
        # Остановка процедуры
        if np.max(np.abs(polyhedron[best] - polyhedron[worst])) <= e:
            return polyhedron[best], values[best], count
        else:
            penalty_param = k


# Генерация начального многоугольника
def generate_initial_polyhedron(initial_point):
    n = len(initial_point)
    polyhedron = [initial_point]
    for i in range(n):
        point = initial_point.copy()
        point[i] += 1.0
        polyhedron.append(point)
    return np.array(polyhedron, dtype=float)

def main():
    e = math.pow(10, -6)

    # Начальное значение штрафного параметра
    penalty_param = 1

    initial_point = [1, 1]
    minimum, min_value, count = nelder_mead(penalty_function, initial_point, penalty_param)

    print("Минимум найден в точке:", minimum)
    print("Значение минимума:", min_value)
    print("Количество итераций: ", count)


if __name__ == "__main__":
    main()
