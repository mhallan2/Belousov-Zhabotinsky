import numpy as np

def ode_system_1(t, y):
    """Система 1: Модель Белоусова-Жаботинского"""
    y1, y2, y3 = y
    dy1dt = 77.27 * (y2 + y1 * (1 - 8.375e-6 * y1 - y2))
    dy2dt = (1 / 77.27) * (y3 - (1 + y1) * y2)
    dy3dt = 0.16 * (y1 - y3)
    return np.array([dy1dt, dy2dt, dy3dt])

def ode_system_2(t, y):
    """Система 2: Химическая кинетика"""
    y1, y2, y3 = y
    dy1dt = -0.04 * y1 + 1e4 * y2 * y3
    dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3dt = 3e7 * y2**2
    return np.array([dy1dt, dy2dt, dy3dt])

def ode_system_3(t, y, A=7.89e-10, B=1.1e7, C=1.13e3, M=1e6):
    """Система 3: Расширенная химическая модель"""
    y1, y2, y3, y4 = y
    dy1dt = -A * y1 - B * y1 * y3
    dy2dt = A * y1 - M * C * y2 * y3
    dy3dt = A * y1 - B * y1 * y3 - M * C * y2 * y3 + C * y4
    dy4dt = B * y1 * y3 - C * y4
    return np.array([dy1dt, dy2dt, dy3dt, dy4dt])

# Параметры для каждой системы
systems = {
    'system1': {
        'function': ode_system_1,
        'y0': np.array([1.0, 2.0, 3.0]),
        't_span': (0.0, 800.0),
        'ylims': [(0.0, 125e3), (0.0, 2e3), (0.0, 35e3)],
        'colors': ['r', 'g', 'b']
    },
    'system2': {
        'function': ode_system_2,
        'y0': np.array([1.0, 0.0, 0.0]),
        't_span': (0.0, 1000.0),
        'ylims': [(0.0, 1.0), (0.0, 5e-5), (0.0, 1.0)],
        'colors': ['m', 'c', 'y']
    },
    'system3': {
        'function': ode_system_3,
        'y0': np.array([1.76e-3, 0.0, 0.0, 0.0]),
        't_span': (0.0, 1013.0),
        'ylims': [(15e-4, 20e-4), (0.0, 2e-10), (0.0, 1e-11), (0.0, 2e-10)],
        'colors': ['r', 'g', 'b', 'k']
    }
}
