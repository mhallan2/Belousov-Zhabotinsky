import numpy as np
import time
import matplotlib.pyplot as plt
from implicit_rk4 import implicit_rk4


def ode_system_3(t, y, A=7.89e-10, B=1.1e7, C=1.13e3, M=1e6):
    y1, y2, y3, y4 = y

    dy1dt = -A * y1 - B * y1 * y3
    dy2dt = A * y1 - M * C * y2 * y3
    dy3dt = A * y1 - B * y1 * y3 - M * C * y2 * y3 + C * y4
    dy4dt = B * y1 * y3 - C * y4

    return np.array([dy1dt, dy2dt, dy3dt, dy4dt])


# Параметры системы
y0 = np.array([1.76e-3, 0.0, 0.0, 0.0])
t_span = (0.0, 1013.0)
y_lims_234 = (0.0, 2e-10) # для y2, y3, y4
y_lims_1 = (15e-4, 20e-4)  # для y1

if __name__ == "__main__":
    start_time = time.time()

    h = 0.025  # 0.025 работает быстро
    t, Y = implicit_rk4(ode_system_3, t_span, y0, h)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)

    plt.plot(t, Y[:, 0], 'r', label=r'$y_1(t)$')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('(3) Численное решение для y1(t)')
    plt.ylim(y_lims_1)
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)

    plt.plot(t, Y[:, 1], 'g', label=r'$y_2(t)$')
    plt.plot(t, Y[:, 2], 'b', label=r'$y_3(t)$')
    plt.plot(t, Y[:, 3], 'k', label=r'$y_4(t)$')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('(3) Численное решение для y2(t), y3(t), y4(t)')
    plt.ylim(y_lims_234)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    plt.show()
