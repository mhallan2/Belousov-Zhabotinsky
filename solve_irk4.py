import numpy as np
from numpy.linalg import solve, norm
from typing import Callable, Tuple, Optional, List
import matplotlib.pyplot as plt
from ode_system_1 import ode_system_1, jacobian_1

# -------------------------------------------------------------------------
# Gauss–Legendre 2‑stage (order 4) Butcher tableau
# -------------------------------------------------------------------------
sqrt3 = np.sqrt(3.0)
c1 = 0.5 - sqrt3 / 6.0
c2 = 0.5 + sqrt3 / 6.0
A = np.array([[0.25, 0.25 - sqrt3 / 6.0], [0.25 + sqrt3 / 6.0, 0.25]])
b = np.array([0.5, 0.5])


# -------------------------------------------------------------------------
# Solver: 2‑stage Radau IIA (Gauss–Legendre) with adaptive step
# -------------------------------------------------------------------------
def solve_irk4(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    jac: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    atol: float = 1e-13,
    rtol: float = 1e-11,
    h0: Optional[float] = None,
    h_min: float = 1e-12,
    h_max: float = 1e-8,
    newton_tol: float = 1e-10,
    newton_maxiter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with implicit RK4.

    Returns
    -------
    t : (N,) array of time points
    y : (N, m) array of solution values
    """
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span must satisfy t_span[1] > t_span[0]")

    y0 = np.asarray(y0, float)
    n = y0.size
    if h_max is None:
        h_max = abs(tf - t0)

    # Jacobian via finite differences
    def jac_fd(ti, yi, fi=None, eps=1e-10):
        fi = f(ti, yi) if fi is None else fi
        J = np.empty((n, n), float)
        for j in range(n):
            dy = np.zeros_like(yi)
            dy[j] = eps * (1 + abs(yi[j]))
            J[:, j] = (f(ti, yi + dy) - fi) / dy[j]
        return J

    # Solve two-stage system with Newton
    def solve_stages(ti, yi, h):
        # initial guess: forward eval
        K = np.tile(f(ti, yi), (2, 1))
        for _ in range(newton_maxiter):
            # stage values
            Y1 = yi + h * (A[0, 0] * K[0] + A[0, 1] * K[1])
            Y2 = yi + h * (A[1, 0] * K[0] + A[1, 1] * K[1])
            F1 = f(ti + c1 * h, Y1)
            F2 = f(ti + c2 * h, Y2)

            # residual
            R1 = K[0] - F1
            R2 = K[1] - F2
            R = np.hstack([R1, R2])
            if norm(R, np.inf) < newton_tol:
                return K

            # Jacobians
            J1 = jac(ti + c1 * h, Y1) if jac else jac_fd(ti + c1 * h, Y1, F1)
            J2 = jac(ti + c2 * h, Y2) if jac else jac_fd(ti + c2 * h, Y2, F2)

            I = np.eye(n)
            # build block Jacobian
            Jblk = np.block(
                [
                    [I - h * A[0, 0] * J1, -h * A[0, 1] * J1],
                    [-h * A[1, 0] * J2, I - h * A[1, 1] * J2],
                ]
            )
            delta = solve(Jblk, -R)
            K += delta.reshape(2, n)
        raise RuntimeError("Newton failed to converge")

    # error norm: infinity norm scaled
    def err_norm(e, sc):
        return np.max(np.abs(e / sc))

    # initial step
    if h0 is None:
        f0 = f(t0, y0)
        scale0 = atol + np.abs(y0) * rtol
        h0 = 0.01 * norm(scale0) / max(norm(f0), 1e-8)
    h = np.clip(h0, h_min, h_max)

    t_vals: List[float] = [t0]
    y_vals: List[np.ndarray] = [y0]
    safety, fac_min, fac_max, p = 0.8, 0.5, 2.0, 4.0

    while t_vals[-1] < tf:
        ti, yi = t_vals[-1], y_vals[-1]
        h = min(h, tf - ti)

        # full step
        try:
            K = solve_stages(ti, yi, h)
        except RuntimeError:
            h *= 0.5
            if h < h_min:
                raise RuntimeError("Couldn't calculate K (step is too small)")
            continue
        y_full = yi + h * (b[0] * K[0] + b[1] * K[1])

        # two half-steps
        h2 = 0.5 * h
        try:
            K1 = solve_stages(ti, yi, h2)
            y_half = yi + h2 * (b[0] * K1[0] + b[1] * K1[1])
            K2 = solve_stages(ti + h2, y_half, h2)
        except RuntimeError:
            h *= 0.5
            if h < h_min:
                raise
            continue
        y_two = y_half + h2 * (b[0] * K2[0] + b[1] * K2[1])

        # error estimate
        e = y_two - y_full
        scale = atol + np.maximum(np.abs(y_two), np.abs(y_full)) * rtol
        rho = err_norm(e, scale)
        rho = max(rho, 1e-16)

        if rho <= 1.0:
            # accept
            t_vals.append(ti + h)
            y_vals.append(y_two)
            # update step
            h *= min(fac_max, max(fac_min, safety * rho ** (-1.0 / (p + 1.0))))
        else:
            # reject
            h *= max(fac_min, safety * rho ** (-1.0 / (p + 1.0)))
            if h < h_min:
                raise RuntimeError("Step size too small")

    return np.array(t_vals), np.vstack(y_vals)


def jacobian_test(t, y):
    # производная по y: [[-1,0,0],[1,-1,0],[0,1,-2]]
    return np.array([[-1, 0, 0], [1, -1, 0], [0, 1, -2]])


# Система ОДУ с известным аналитическим решением
def ode_system_test(t, y):
    y1, y2, y3 = y
    dy1dt = -y1
    dy2dt = y1 - y2
    dy3dt = y2 - 2 * y3
    return np.array([dy1dt, dy2dt, dy3dt])

t_span_test = (0.0, 1000.0)
y0_test = np.array([1.0, 0.0, 0.0])

# Аналитическое решение (возвращает массив формы (n_points, 3))
def exact_solution(t):
    y1 = np.exp(-t)
    y2 = np.exp(-t) * t
    y3 = np.exp(-2 * t) * (np.exp(t) * (t - 1) + 1)
    return np.column_stack([y1, y2, y3])


def plot(ode, t_span, y0, jac, h=1e-2):
    t_num, y_num = solve_irk4(ode, t_span, y0, jac, h0=h)
    # y_ex = exact_solution(t_num)
    plt.figure(figsize=(12, 10))
    plt.plot(t_num, y_num, label="numerical solution")
    # plt.plot(t_num, y_ex, '--', label='exact solution')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Параметры решения
    h = 1e-2
    plot(ode_system_test, t_span_test, y0_test, jacobian_1, h)
