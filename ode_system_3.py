import numpy as np
import time
import matplotlib.pyplot as plt
from implicit_rk4 import solve_irk4


def ode_system_3(t, y, A=7.89e-10, B=1.1e7, C=1.13e3, M=1e6):
    y1, y2, y3, y4 = y

    dy1dt = -A * y1 - B * y1 * y3
    dy2dt = A * y1 - M * C * y2 * y3
    dy3dt = A * y1 - B * y1 * y3 - M * C * y2 * y3 + C * y4
    dy4dt = B * y1 * y3 - C * y4

    return np.array([dy1dt, dy2dt, dy3dt, dy4dt])


# Параметры системы
y0 = np.array([1.76e-3, 0.0, 0.0, 0.0])
t_span = (0.0, 1013.0)
y_lims = (0.0, 10e-10)

if __name__ == "__main__":
    start_time = time.time()

    h = 0.025  # 0.025 работает?
    t, Y = solve_irk4(ode_system_3, t_span, y0, h)

    # визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(t, Y[0], 'r', label=r'$y_1(t)$')
    plt.plot(t, Y[1], 'g', label=r'$y_2(t)$')
    plt.plot(t, Y[2], 'b', label=r'$y_3(t)$')
    plt.xlabel('Время $t$')
    plt.ylabel('Решение $y(t)$')
    plt.title('Неявный метод Рунге–Кутта 4-го порядка (3-ья система)')
    plt.xlim(t[0], t[-1])
    plt.ylim(*y_lims)
    plt.legend()
    plt.grid()
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    plt.show()
