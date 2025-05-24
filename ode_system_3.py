import numpy as np


def ode_system_3(t, y, A=7.89e-10, B=1.1e7, C=1.13e3, M=1e6):
    y1, y2, y3, y4 = y

    dy1dt = -A * y1 - B * y1 * y3
    dy2dt = A * y1 - M * C * y2 * y3
    dy3dt = A * y1 - B * y1 * y3 - M * C * y2 * y3 + C * y4
    dy4dt = B * y1 * y3 - C * y4

    return np.array([dy1dt, dy2dt, dy3dt, dy4dt])


def jacobian_3(t, y, A=7.89e-10, B=1.1e7, C=1.13e3, M=1e6):
    y1, y2, y3, y4 = y

    # Производные для dy1dt
    df1_dy1 = -A - B * y3
    df1_dy2 = 0.0
    df1_dy3 = -B * y1
    df1_dy4 = 0.0

    # Производные для dy2dt
    df2_dy1 = A
    df2_dy2 = -M * C * y3
    df2_dy3 = -M * C * y2
    df2_dy4 = 0.0

    # Производные для dy3dt
    df3_dy1 = A - B * y3
    df3_dy2 = -M * C * y3
    df3_dy3 = -B * y1 - M * C * y2
    df3_dy4 = C

    # Производные для dy4dt
    df4_dy1 = B * y3
    df4_dy2 = 0.0
    df4_dy3 = B * y1
    df4_dy4 = -C

    J = np.array(
        [
            [df1_dy1, df1_dy2, df1_dy3, df1_dy4],
            [df2_dy1, df2_dy2, df2_dy3, df2_dy4],
            [df3_dy1, df3_dy2, df3_dy3, df3_dy4],
            [df4_dy1, df4_dy2, df4_dy3, df4_dy4],
        ]
    )
    return J


y0_3 = np.array([1.76e-3, 0.0, 0.0, 0.0])
t_span_3 = (0.0, 1013.0)
