import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def ode_system(t, y):
    """
    Система ОДУ:
    dy1/dt = 77.27 [y2 + y1 (1 - 8.375e-6 y1 - y2)],
    dy2/dt = (1/77.27) [y3 - (1 + y1) y2],
    dy3/dt = 0.16 (y1 - y3).
    """
    y1, y2, y3 = y
    dy1dt = 77.27 * (y2 + y1 * (1 - 8.375e-6 * y1 - y2))
    dy2dt = (1 / 77.27) * (y3 - (1 + y1) * y2)
    dy3dt = 0.16 * (y1 - y3)
    return np.array([dy1dt, dy2dt, dy3dt])


start_time = time.time()
# Параметры решения
h = 10e-2
start = 0.0
end = 800.0
N = int((end - start) / h)
y0 = np.array([1.0, 2.0, 3.0])

# Используем solve_ivp с методом RK45 (явный метод Рунге-Кутты 4-го порядка)
sol = solve_ivp(ode_system, [start, end], y0, method='Radau', t_eval=np.linspace(start, end, N))

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], 'r', label='$y_1(t)$')
plt.plot(sol.t, sol.y[1], 'g', label='$y_2(t)$')
plt.plot(sol.t, sol.y[2], 'b', label='$y_3(t)$')
plt.xlabel('Время $t$')
plt.ylabel('Решение $y(t)$')
plt.title('Решение системы ОДУ методом Рунге-Кутты 4-го порядка (solve_ivp)')
plt.xlim(start, end)  # Ось X от 0 до 5
plt.ylim(0.0, 125000.0)  # Ось Y от 0 до 15
plt.legend()
plt.grid()
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
