import numpy as np


# Система ОДУ
def ode_system_2(t, y):
    y1, y2, y3 = y
    dy1dt = -0.04 * y1 + 1e4 * y2 * y3
    dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3dt = 3e7 * y2**2
    return np.array([dy1dt, dy2dt, dy3dt])


def jacobian_2(t, y):
    y1, y2, y3 = y

    df1_dy1 = -0.04
    df1_dy2 = 1e4 * y3
    df1_dy3 = 1e4 * y2

    df2_dy1 = 0.04
    df2_dy2 = -1e4 * y3 - 6e7 * y2  # производная от -3e7*y2² = -6e7*y2
    df2_dy3 = -1e4 * y2

    df3_dy1 = 0.0
    df3_dy2 = 6e7 * y2  # производная от 3e7*y2² = 6e7*y2
    df3_dy3 = 0.0

    J = np.array(
        [
            [df1_dy1, df1_dy2, df1_dy3],
            [df2_dy1, df2_dy2, df2_dy3],
            [df3_dy1, df3_dy2, df3_dy3],
        ]
    )
    return J


y0_2 = np.array([1.0, 0.0, 0.0])
t_span_2 = (0.0, 100.0)

