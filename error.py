import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from solve_irk4 import solve_irk4, ode_system_test, jacobian_test, t_span_test, y0_test
from ode_system_1 import ode_system_1, jacobian_1, y0_1, t_span_1
from ode_system_2 import ode_system_2, jacobian_2, y0_2, t_span_2
from ode_system_3 import ode_system_3, jacobian_3, y0_3, t_span_3


# --- Настройка ---
plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "DejaVu Sans"

problems = [
    (ode_system_1, t_span_1, jacobian_1, y0_1, "System 1"),
    #(ode_system_2, t_span_2, jacobian_2, y0_2, "System 2"),
    #(ode_system_3, t_span_3, jacobian_3, y0_3, "System 3"),
    #(ode_system_test, t_span_test, jacobian_test, y0_test, "System test"),
]

tols = np.geomspace(1e-10, 1e-2, num=200)
for ode, t_span, jac, y0, label in problems:
    t0, tf = t_span
    dim = y0.size

    # эталон
    sol_ref = solve_ivp(
        ode, t_span, y0, method="Radau", atol=1e-10, rtol=1e-12, jac=jac
    )
    t_ref = sol_ref.t
    y_ref = sol_ref.y.T

    # подготовка массивов
    comp_errors = np.zeros((len(tols), dim))
    max_errors = np.zeros(len(tols))

    all_t_num = []
    for k, atol in enumerate(tols):
        rtol = 1e-2 * atol  # или atol
        try:
            t_num, y_num = solve_irk4(
                f=ode, t_span=(t0, tf), y0=y0, jac=jac, atol=atol, rtol=rtol
            )
            all_t_num.append(t_num)
            # кубическая интерполяция на t_ref
            interp = interp1d(
                t_num, y_num.T, kind="cubic", axis=1, fill_value="extrapolate"
            )
            y_i = interp(t_ref).T  # (Nt_ref, dim)

            # по компонентам
            comp_errors[k, :] = np.max(np.abs(y_i - y_ref), axis=0)
            # и максимальная (возможно, не стоит смотреть на эту характеристику)
            max_errors[k] = comp_errors[k, :].max()
            print(comp_errors[k, :])
        except Exception as e:
            #print(f"[{label}] Ошибка при atol={atol:.1e}: {e}")
            print(comp_errors[k, :])
            comp_errors[k, :] = np.nan
            max_errors[k] = np.nan

    # убираем nan
    valid = ~np.isnan(max_errors)
    atols_used = tols[valid]
    max_err = max_errors[valid]
    comp_err = comp_errors[valid, :]

    # регрессия на max_errors
    log_a = np.log10(atols_used)
    log_e = np.log10(max_err)
    (slope, intercept), V = np.polyfit(log_a, log_e, 1, cov=True)
    rcoef = np.corrcoef(log_a, log_e)[0, 1]
    p = 4.0
    C = max_err[0] / atols_used[0] ** (p / (p + 1))

    # --- Рисунок с тремя компонентами + max ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    # Подготовили series и titles, как раньше:
    series = [comp_err[:, i] for i in range(dim)] + [max_err]
    titles = [f"Компонента {i + 1}" for i in range(dim)] + [
        "Max по компонентам"
    ]

    component_slopes = []
    for ax, errs, title in zip(axes, series, titles):
        # логарифмы для этой серии
        log_errs = np.log10(errs)

        # своя регрессия
        slope_i, intercept_i = np.polyfit(log_a, log_errs, 1)
        component_slopes.append(slope_i)
        # можно также получить cov-матрицу, R² и т.д.
        rcoef = np.corrcoef(log_a, log_e)[0, 1]
        # свои fit‐точки
        fit_line = 10 ** (slope_i * log_a + intercept_i)

        ax.loglog(atols_used, errs, "o", label="эксп.")
        ax.loglog(atols_used, fit_line, "--", label=f"fit slope={slope_i:.2f}\n"
                                                    f"R**2 = {rcoef ** 2:.2f}")
        ax.loglog(
            atols_used,
            C * atols_used ** (p / (p + 1)),
            ":",
            label=f"theo p/(p+1)={p / (p + 1):.2f}",
        )

        ax.set_title(title)
        ax.set_xlabel("atol")
        ax.set_ylabel("error")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.suptitle(f"Сходимость IRK4 — {label}")
    plt.tight_layout()
    plt.show()
    print(
        f"[{label}] slope={slope:.3f} ±{np.sqrt(V[0, 0]):.3f}, R²={rcoef**2:.3f}"
    )
for i, s in enumerate(component_slopes):
    print(f"Компонента {i+1}: slope = {s:.4f}")
