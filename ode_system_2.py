import numpy as np
import time
import matplotlib.pyplot as plt
from implicit_rk4 import solve_irk4


# Система ОДУ
def ode_system_2(t, y):
    y1, y2, y3 = y
    dy1dt = -0.04 * y1 + 1e4 * y2 * y3
    dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3dt = 3e7 * y2**2
    return np.array([dy1dt, dy2dt, dy3dt])


y0 = np.array([1.0, 0.0, 0.0])
t_span = (0.0, 1000.0)

if __name__ == "__main__":
    start_time = time.time()

    h = 0.025  # 0.025 работает?
    t, Y = solve_irk4(ode_system_2, t_span, y0, h)

    # визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(t, Y[0], 'r', label=r'$y_1(t)$')
    plt.plot(t, Y[1], 'g', label=r'$y_2(t)$')
    plt.plot(t, Y[2], 'b', label=r'$y_3(t)$')
    plt.xlabel('Время $t$')
    plt.ylabel('Решение $y(t)$')
    plt.title('Неявный метод Рунге–Кутта 4-го порядка (2-ая система)')
    plt.xlim(t[0], t[-1])
    plt.ylim(0.0, 2.0)
    plt.legend()
    plt.grid()
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    plt.show()
