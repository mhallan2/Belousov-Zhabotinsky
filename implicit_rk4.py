import numpy as np
from scipy.optimize import fsolve


def implicit_rk4(f, t_span, y0, h, tol=1e-6, maxiter=100):
    # Коэффициенты метода Гаусса 4-го порядка
    sqrt3 = np.sqrt(3)
    c = np.array([0.5 - sqrt3 / 6.0, 0.5 + sqrt3 / 6.0])
    A = np.array([[0.25, 0.25 - sqrt3 / 6.0],
                  [0.25 + sqrt3 / 6.0, 0.25]])
    b = np.array([0.5, 0.5])

    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n_steps = len(t)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    k1_prev, k2_prev = f(t0, y0), f(t0, y0)  # Для начального приближения

    for i in range(n_steps - 1):
        t_curr = t[i]
        y_curr = y[i]

        def F(k):
            k1, k2 = k[:len(y0)], k[len(y0):]
            y1 = y_curr + h * (A[0, 0] * k1 + A[0, 1] * k2)
            y2 = y_curr + h * (A[1, 0] * k1 + A[1, 1] * k2)
            f1 = f(t_curr + c[0] * h, y1)
            f2 = f(t_curr + c[1] * h, y2)
            return np.concatenate([k1 - f1, k2 - f2])

        # Начальное приближение из предыдущего шага
        k0 = np.concatenate([k1_prev, k2_prev])

        k_sol = fsolve(F, k0, xtol=tol, maxfev=maxiter)
        k1, k2 = k_sol[:len(y0)], k_sol[len(y0):]
        k1_prev, k2_prev = k1, k2  # Запоминаем для следующего шага

        y[i + 1] = y_curr + h * (b[0] * k1 + b[1] * k2)

    return t, y
