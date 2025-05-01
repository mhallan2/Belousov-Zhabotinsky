import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ode_system_1 import ode_system_1
from ode_system_2 import ode_system_2
from ode_system_3 import ode_system_3

start_time = time.time()
# Параметры решения
h = 10e-2
start = 0.0
end = 1013.0
N = int((end - start) / h)
y0 = np.array([1.76e-3, 0.0, 0.0, 0.0])

# Используем solve_ivp с методом RK45 (явный метод Рунге-Кутты 4-го порядка)
sol = solve_ivp(ode_system_3, [start, end],
                y0,
                method='Radau',
                t_eval=np.linspace(start, end, N))

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], 'r', label='$y_1(t)$')
plt.plot(sol.t, sol.y[1], 'g', label='$y_2(t)$')
plt.plot(sol.t, sol.y[2], 'b', label='$y_3(t)$')
plt.xlabel('Время $t$')
plt.ylabel('Решение $y(t)$')
plt.title('Решение системы ОДУ методом Рунге-Кутты 4-го порядка (solve_ivp)')
plt.xlim(start, end)  # Ось X от 0 до 5
plt.ylim(0.0, 10e-10)  # Ось Y от 0 до 15
plt.legend()
plt.grid()
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
