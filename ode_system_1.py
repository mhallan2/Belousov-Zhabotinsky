import numpy as np
import time
import matplotlib.pyplot as plt
from implicit_rk4 import implicit_rk4


# Система ОДУ
def ode_system_1(t, y):
    y1, y2, y3 = y
    dy1dt = 77.27 * (y2 + y1 * (1 - 8.375e-6 * y1 - y2))
    dy2dt = (1 / 77.27) * (y3 - (1 + y1) * y2)
    dy3dt = 0.16 * (y1 - y3)
    return np.array([dy1dt, dy2dt, dy3dt])


y0 = np.array([1.0, 2.0, 3.0])
t_span = (0.0, 800.0)

if __name__ == "__main__":
    start_time = time.time()

    h = 0.025  # 0.025 работает быстро
    t, Y = implicit_rk4(ode_system_1, t_span, y0, h)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, Y[:, 0], 'r', label=r'$y_1(t)$')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.ylim(0.0, 125e3)
    plt.title('(1) Численное решение для y1(t)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, Y[:, 1], 'g', label=r'$y_2(t)$')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.ylim(0.0, 2e3)
    plt.title('(1) Численное решение для y2(t)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, Y[:, 2], 'b', label=r'$y_3(t)$')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.ylim(0.0, 35e3)
    plt.title('(1) Численное решение для y3(t)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    plt.show()