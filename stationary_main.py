from error_analysis import stationary_true_solution_tensor, error
from export_utils import export_results
from project import FD_solver, true_sol
from utils import get_linear_index, convert_to_3D
import matplotlib.pyplot as plt

errors = []
steps = []
steps_squared = []
for step in range(5, 20):
    Nx = Ny = Nz = step
    ts = stationary_true_solution_tensor(true_sol, Nx, Ny, Nz)
    u = FD_solver(Nx, Ny, Nz)
    er = error(ts, u)
    ts_tensor = convert_to_3D(ts, Nx, Ny, Nz)
    u_tensor = convert_to_3D(u, Nx, Ny, Nz)
    errors.append(er)
    steps.append(1 / (step + 1))
    steps_squared.append((1 / (step + 1)) ** 2)

export_results('fd_stationary.vti', u_tensor)
export_results('true_stationary.vti', ts_tensor)

plt.title('Error')
plt.xlabel('Steps')
plt.ylabel('Error')

plt.plot(errors, steps, label='Error')
plt.plot(steps_squared, steps, label='h^2')
plt.grid(True)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig('stationary_error_plot.png')
plt.show()
