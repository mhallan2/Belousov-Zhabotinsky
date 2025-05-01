import numpy as np
import matplotlib.pyplot as plt
from ode_system_1 import solve_irk4


# Система ОДУ с известным аналитическим решением
def ode_system(t, y):
    y1, y2, y3 = y
    dy1dt = -y1
    dy2dt = y1 - y2
    dy3dt = y2 - 2 * y3
    return np.array([dy1dt, dy2dt, dy3dt])


# Аналитическое решение
def exact_solution(t):
    y1 = np.exp(-t)
    y2 = np.exp(-t) * t
    y3 = np.exp(-2 * t) * (np.exp(t) * (t - 1) + 1)
    return np.array([y1, y2, y3])


# Параметры решения
t_span = (0.0, 10.0)
y0 = np.array([1.0, 0.0, 0.0])
h = 0.1

# Численное и точное решения
t, Y_num = solve_irk4(ode_system, t_span, y0, h)
Y_exact = exact_solution(t)

# Ошибка
error = np.abs(Y_num - Y_exact)

# Визуализация
plt.figure(figsize=(14, 10))

# Графики решений
plt.subplot(2, 1, 1)
plt.plot(t, Y_num[0], 'r--', label='Численное $y_1(t)$')
plt.plot(t, Y_num[1], 'g--', label='Численное $y_2(t)$')
plt.plot(t, Y_num[2], 'b--', label='Численное $y_3(t)$')
plt.plot(t, Y_exact[0], 'k-', alpha=0.9, label='Точное $y_1(t)$')
plt.plot(t, Y_exact[1], 'k-', alpha=0.5, label='Точное $y_2(t)$')
plt.plot(t, Y_exact[2], 'k-', alpha=0.3, label='Точное $y_3(t)$')
plt.xlabel('Время $t$')
plt.ylabel('$y(t)$')
plt.title('Сравнение численного и точного решений')
plt.legend()
plt.grid()

# График ошибки в логарифмических координатах с аппроксимацией
plt.subplot(2, 1, 2)
log_t = np.log(t[1:])  # Исключаем t=0, так как log(0) не определен
log_error = np.log(error[:, 1:])  # Соответственно исключаем первую точку

# Аппроксимация линейной зависимостью для каждой компоненты
for i, color in enumerate(['r', 'g', 'b']):
    coeffs = np.polyfit(log_t, log_error[i], 1)
    slope = coeffs[0]
    plt.plot(log_t,
             np.polyval(coeffs, log_t),
             color + '--',
             label=f'Ошибка $y_{i+1}$ (наклон: {slope:.2f})')

plt.xlabel('$\ln(t)$')
plt.ylabel('$\ln$(ошибки)')
plt.title('Логарифм ошибки с линейной аппроксимацией')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Проверка порядка метода
hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]
errors = []
for h in hs:
    t, Y_num = solve_irk4(ode_system, (0.0, 1.0), y0, h)
    Y_exact = exact_solution(t)
    errors.append(np.max(np.abs(Y_num[:, -1] - Y_exact[:, -1])))

# Оценка порядка
orders = [
    float(np.log(errors[i] / errors[i + 1]) / np.log(2))
    for i in range(len(errors) - 1)
]
print("Ошибки при h =", hs)
print("Оценки порядка метода:", orders)
