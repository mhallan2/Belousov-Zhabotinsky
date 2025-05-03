import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ode_system_1 import ode_system_1 as sys1, y0 as y0_1, t_span as t_span_1
from ode_system_2 import ode_system_2 as sys2, y0 as y0_2, t_span as t_span_2
from ode_system_3 import ode_system_3 as sys3, y0 as y0_3, t_span as t_span_3

start_time = time.time()
# Параметры решения
h = 10e-2
start, end = t_span_2
N = int((end - start) / h)
sys = sys2
y0 = y0_2

# Используем solve_ivp с методом RK45 (явный метод Рунге-Кутты 5-го порядка)
sol = solve_ivp(sys, [start, end],
                y0,
                method='Radau',
                t_eval=np.linspace(start, end, N))

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], 'r', label='$y_1(t)$')
plt.plot(sol.t, sol.y[1], 'g', label='$y_2(t)$')
plt.plot(sol.t, sol.y[2], 'b', label='$y_3(t)$')
try:
    plt.plot(sol.t, sol.y[3], 'k', label='$y_4(t)$')
except IndexError:
    pass
plt.xlabel('Время $t$')
plt.ylabel('Решение $y(t)$')
plt.title('Решение системы ОДУ методом Рунге-Кутты 4-го порядка (solve_ivp)')
plt.legend()
plt.grid()
plt.tight_layout()
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
