import numpy as np


# Система ОДУ
def ode_system_1(t, y):
    y1, y2, y3 = y
    dy1dt = 77.27 * (y2 + y1 * (1 - 8.375e-6 * y1 - y2))
    dy2dt = (1 / 77.27) * (y3 - (1 + y1) * y2)
    dy3dt = 0.16 * (y1 - y3)
    return np.array([dy1dt, dy2dt, dy3dt])


def jacobian_1(t, y):
    y1, y2, y3 = y

    df1_dy1 = 77.27 * (1 - 1.675e-5 * y1 - y2)
    df1_dy2 = 77.27 * (1 - y1)
    df1_dy3 = 0.0

    df2_dy1 = -y2 / 77.27
    df2_dy2 = -(1 + y1) / 77.27
    df2_dy3 = 1 / 77.27

    df3_dy1 = 0.16
    df3_dy2 = 0.0
    df3_dy3 = -0.16

    J = np.array([
        [df1_dy1, df1_dy2, df1_dy3],
        [df2_dy1, df2_dy2, df2_dy3],
        [df3_dy1, df3_dy2, df3_dy3]
    ])

    return J


y0_1 = np.array([1.0, 2.0, 3.0])
t_span_1 = (0.0, 800.0)
