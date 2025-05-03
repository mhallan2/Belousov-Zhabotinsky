import numpy as np
import time
import matplotlib.pyplot as plt
from implicit_rk4 import implicit_rk4


# Система ОДУ
def ode_system_2(t, y):
    y1, y2, y3 = y
    dy1dt = -0.04 * y1 + 1e4 * y2 * y3
    dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3dt = 3e7 * y2**2
    return np.array([dy1dt, dy2dt, dy3dt])


y0 = np.array([1.0, 0.0, 0.0])
t_span = (0.0, 1000.0)
y_lims = (0.0, 1.0)

if __name__ == "__main__":
    start_time = time.time()

    h = 0.025  # 0.025 работает быстро
    t, Y = implicit_rk4(ode_system_2, t_span, y0, h)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(t, Y[:, 0], 'r', label=r'$y_1(t)$')
    plt.plot(t, Y[:, 2], 'b', label=r'$y_3(t)$')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.ylim(0.0, 1.0)
    plt.title('(1) Численное решение для y1(t)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, Y[:, 1], 'g', label=r'$y_2(t)$')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.ylim(0.0, 5e-5)
    plt.title('(1) Численное решение для y2(t)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    plt.show()
