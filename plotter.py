import time
import matplotlib.pyplot as plt
from implicit_rk4 import implicit_rk4
from systems import ode_system_1, ode_system_2, ode_system_3, systems


def plot_system(system_func, params, h=0.025):
    """Универсальная функция для построения графиков"""
    start_time = time.time()

    # Решаем систему
    t, Y = implicit_rk4(system_func, params['t_span'], params['y0'], h)

    # Создаем графики
    plt.figure(figsize=(15, 11))

    for i in range(Y.shape[1]):
        plt.subplot(Y.shape[1], 1, i + 1)
        plt.plot(t, Y[:, i], params['colors'][i], label=f'$y_{i + 1}(t)$')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.ylim(params['ylims'][i])
        plt.title(f'Численное решение для $y_{i + 1}(t)$')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    plt.show()


if __name__ == "__main__":
    systems_funcs = [ode_system_1, ode_system_2, ode_system_3]
    [plot_system(f, systems[f'system{i+1}']) for i, f in enumerate(systems_funcs)]