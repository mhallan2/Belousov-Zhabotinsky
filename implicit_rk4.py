import numpy as np
from scipy.optimize import root


# Шаг по Гауссу-Лежандру (2-ух ступенчатый, 4-ого порядка)
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
