import numpy as np
from time_dependent import td_true_solution_tensors, theta_method, true_sol
import matplotlib.pyplot as plt
from export_utils import export_time_dependent

N = 10
M = 20

errors = []
steps = []
steps_squared = []
for step in range(5, 20):
    M = step
    ut = theta_method(0.5, 1, N, M)
    u_true = td_true_solution_tensors(true_sol, N, N, N, M)
    er = np.linalg.norm(ut[-1] - u_true[-1])
    errors.append(er)
    steps.append(1 / (step + 1))
    steps_squared.append((1 / (step + 1))**2)

N = 10
M = 20
ut = theta_method(0.5, 1, N, M)
export_time_dependent(ut, 'tds_theta', N, './td_theta_solution')

u_true = td_true_solution_tensors(true_sol, N, N, N, M)
export_time_dependent(u_true, 'tds_true', N, './td_true_solution')

plt.title('Error')
plt.xlabel('Steps')
plt.ylabel('Error')

plt.plot(errors, steps, label='Error')
plt.plot(steps_squared, steps, label='h^2')
plt.plot(steps, steps, label='h')
plt.grid(True)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig('td_error_plot.png')
plt.show()
