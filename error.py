import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline, interp1d
from solve_irk4 import solve_irk4, ode_system_test, jacobian_test, t_span_test, y0_test
from ode_system_1 import ode_system_1, jacobian_1, y0_1, t_span_1
from ode_system_2 import ode_system_2, jacobian_2, y0_2, t_span_2
from ode_system_3 import ode_system_3, jacobian_3, y0_3, t_span_3


# --- Настройка ---
plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "DejaVu Sans"

problems = [
    #(ode_system_1, t_span_1, jacobian_1, y0_1, "System 1"),
    #(ode_system_2, t_span_2, jacobian_2, y0_2, "System 2"),
    (ode_system_3, t_span_3, jacobian_3, y0_3, "System 3"),
    #(ode_system_test, t_span_test, jacobian_test, y0_test, "System test"),
]

tols = np.geomspace(1e-13, 1e-14, num=20)
for ode, t_span, jac, y0, label in problems:
    start = time.time()

    t0, tf = t_span
    dim = y0.size

    # эталон
    sol_ref = solve_ivp(
        ode, t_span, y0, method="Radau", atol=1e-14, rtol=1e-10, jac=jac
    )
    t_ref = sol_ref.t
    y_ref = sol_ref.y.T

    #t_ref, y_ref = solve_irk4(
    #    f=ode, t_span=(t0, tf), y0=y0, jac=jac, atol=1e-12, rtol=1e-14
    #)

    # подготовка массивов
    comp_errors = np.zeros((len(tols), dim))
    max_errors = np.zeros(len(tols))

    all_t_num = []
    for k, atol in enumerate(tols):
        rtol = 1e+2 * atol  # или atol
        try:
            #sol_num = solve_ivp(
            #    f=ode, t_span=(t0, tf), y0=y0, method="RK45", atol=atol, rtol=rtol
            #)
            #t_num, y_num = sol_num.t, sol_num.y.T

            t_num, y_num = solve_irk4(
                f=ode, t_span=(t0, tf), y0=y0, jac=jac, atol=atol, rtol=rtol
            )

            all_t_num.append(t_num)
            # кубическая интерполяция на t_ref
            interp = interp1d(
                t_num, y_num.T, kind="cubic", axis=1, fill_value="extrapolate"
            )
            y_i = interp(t_ref).T  # (Nt_ref, dim)

            # y_num.shape = (N, dim), нужно интерполировать по каждой компоненте отдельно
            #y_i = np.zeros((len(t_ref), y_num.shape[1]))  # (Nt_ref, dim)

            #for j in range(y_num.shape[1]):  # по каждой компоненте
            #    spline = make_interp_spline(t_num, y_num[:, j], k=3)
            #    y_i[:, j] = spline(t_ref)

            # по компонентам
            comp_errors[k, :] = np.max(np.abs(y_i - y_ref), axis=0)
            # и максимальная (возможно, не стоит смотреть на эту характеристику)
            #max_errors[k] = comp_errors[k, :].max()
            print(comp_errors[k, :])
        except Exception as e:
            #print(f"[{label}] Ошибка при atol={atol:.1e}: {e}")
            #print(comp_errors[k, :])
            comp_errors[k, :] = np.nan
            max_errors[k] = np.nan

    # убираем nan
    valid = ~np.isnan(max_errors)
    atols_used = tols[valid]
    comp_err = comp_errors[valid, :]

    log_a = np.log10(atols_used)
    p = 4.0
    p_log = 0.8

    # --- Рисунок с тремя/четырьмя компонентами ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    series = [comp_err[:, i] for i in range(dim)]
    titles = [f"Компонента {i + 1}" for i in range(dim)]

    component_slopes = []
    for ax, errs, title in zip(axes, series, titles):
        # логарифмы для этой серии
        log_errs = np.log10(errs)

        slope_i, intercept_i = np.polyfit(log_a, log_errs, 1)
        component_slopes.append(slope_i)

        rcoef = np.corrcoef(log_a, log_errs)[0, 1]

        fit_line = 10 ** (slope_i * log_a + intercept_i)
        C = errs[0] / atols_used[0] ** p_log

        ax.loglog(atols_used, errs, "o", label="Experimental")
        ax.loglog(atols_used, fit_line, "--", label=f"Slope={slope_i:.4f}\n"
                                                    f"R**2 = {rcoef ** 2:.4f}")
        ax.loglog(
            atols_used,
            C * atols_used ** (p / (p + 1)),
            ":",
            label=f"Theoretical p/(p+1)=0.8",
        )

        ax.set_title(title)
        ax.set_xlabel("atol")
        ax.set_ylabel("error")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.suptitle(f"Сходимость IRK4 — {label}")
    plt.tight_layout()
    plt.show()

print(f"---- {(time.time() - start):.4f} ----")

for i, s in enumerate(component_slopes):
    print(f"Компонента {i+1}: slope = {s:.4f}")
