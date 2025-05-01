import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import root


# систему ОДУ
def ode_system(t, y):
    y1, y2, y3 = y
    dy1dt = 77.27 * (y2 + y1 * (1 - 8.375e-6 * y1 - y2))
    dy2dt = (1 / 77.27) * (y3 - (1 + y1) * y2)
    dy3dt = 0.16 * (y1 - y3)
    return np.array([dy1dt, dy2dt, dy3dt])


# шаг по Гауссу-Лежандру (2-ух ступенчатый, 4-ого порядка)
def gauss_legendre_step(f, t, y, h):
    sqrt3 = np.sqrt(3.0)
    A = np.array([[0.25, 0.25 - sqrt3 / 6], [0.25 + sqrt3 / 6, 0.25]])
    c = np.array([0.5 - sqrt3 / 6, 0.5 + sqrt3 / 6])
    b = np.array([0.5, 0.5])
    d, s = y.size, 2

    # нелинейная система на стадии K
    def G(K_flat):
        K = K_flat.reshape(s, d)
        Y = y + h * (A @ K)
        F = np.array([f(t + c[i] * h, Y[i]) for i in range(s)])
        return (K - F).ravel()

    k_prev = f(t, y)
    k0 = np.tile(k_prev + 0.01 * h * k_prev, s)
    sol = root(G, k0, method='lm', tol=1e-6)
    if not sol.success:
        print(f"Ошибка на шаге t={t:.3f}, y={y}")
        print(f"Норма невязки: {np.linalg.norm(G(sol.x))}")
        h_new = h / 2.0
        print(h_new)
        return gauss_legendre_step(f, t, y, h_new)

    k = sol.x.reshape(s, d)
    return y + h * (k.T @ b)


# Implicit RK 4
def solve_irk4(f, t_span, y0, h):
    t0, t_end = t_span
    N = int(np.ceil((t_end - t0) / h))
    t = np.empty(N + 1)
    y = np.empty((N + 1, y0.size))

    t[0], y[0] = t0, y0
    for n in range(N):
        t[n + 1] = t[n] + h
        y[n + 1] = gauss_legendre_step(f, t[n], y[n], h)
    return t, y.T


if __name__ == "__main__":
    start_time = time.time()

    h = 0.025  # 0.025 работает?
    t, Y = solve_irk4(ode_system, (0.0, 800.0),
                      np.array([1.0, 2.0, 3.0]), h)

    # визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(t, Y[0], 'r', label=r'$y_1(t)$')
    plt.plot(t, Y[1], 'g', label=r'$y_2(t)$')
    plt.plot(t, Y[2], 'b', label=r'$y_3(t)$')
    plt.xlabel('Время $t$')
    plt.ylabel('Решение $y(t)$')
    plt.title('Неявный метод Рунге–Кутта 4-го порядка (Gauss–Legendre)')
    plt.xlim(t[0], t[-1])
    plt.ylim(0.0, 1.20e5)
    plt.legend()
    plt.grid()
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    plt.show()
