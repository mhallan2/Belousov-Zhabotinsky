import matplotlib.pyplot as plt
from solve_irk4 import solve_irk4
from ode_system_1 import ode_system_1, jacobian_1, y0_1, t_span_1
from ode_system_2 import ode_system_2, jacobian_2, y0_2, t_span_2
from ode_system_3 import ode_system_3, jacobian_3, y0_3, t_span_3
from scipy.integrate import solve_ivp
import time

start = time.time()
t, Y = solve_irk4(ode_system_1, t_span_1, y0_1, jacobian_1, atol=1e-13, rtol=1e-10)
#sol = solve_ivp(ode_system_3, t_span_3, y0_3, method="Radau", jac=jacobian_3, atol=1e-13, rtol=1e-10)
#t, Y = sol.t, sol.y.T
plt.figure(figsize=(12, 8))

plt.plot(Y[:, 1], Y[:, 0], label="y1(y2)")
plt.plot(Y[:, 2], Y[:, 0], label="y1(y3)")
plt.plot(Y[:, 2], Y[:, 1], label="y2(y3)")
# Добавь ещё нужные тебе пары по аналогии

plt.xlabel("x (зависимая переменная)")
plt.ylabel("y (независимая переменная)")
plt.legend()
plt.grid()
plt.ylim(0.0, 0.2*1e-9)
plt.tight_layout()
plt.show()

#plt.figure(figsize=(12,8))
#for i in range(Y.shape[1]):
#    plt.plot(Y[:, (i + 1) % Y.shape[1]], Y[:, i], label=str(i+1))
#    plt.tight_layout()
#    #plt.ylim(0.00150, 0.00185)
#    plt.legend()
#    plt.ylim(0.0, 4*1e-10)
#    plt.grid()
plt.show()
print(time.time() - start)