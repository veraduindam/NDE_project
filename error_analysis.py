from export_utils import export_results
from project import FD_solver, true_sol
from utils import get_linear_index, convert_to_3D
import numpy as np
import matplotlib.pyplot as plt


def error(true, computed):
    return np.linalg.norm(computed - true, ord=np.inf)


def stationary_true_solution_tensor(func, Nx, Ny, Nz):
    hx = 1 / (Nx - 1)
    hy = 1 / (Ny - 1)
    hz = 1 / (Nz - 1)
    true_solution = np.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                linear_index = get_linear_index(i, j, k, Nx, Ny)
                true_solution[linear_index] = func(i * hx, j * hy, k * hz)
    return true_solution


# errors = []
# steps = []
# for step in range(5, 20):
#     Nx = Ny = Nz = step
#     ts = stationary_true_solution_tensor(true_sol, Nx, Ny, Nz)
#     u = FD_solver(Nx, Ny, Nz)
#     er = error(ts, u)
#     ts_tensor = convert_to_3D(ts, Nx, Ny, Nz)
#     u_tensor = convert_to_3D(u, Nx, Ny, Nz)
#     errors.append(er)
#     steps.append(1 / (step + 1))
#
# plt.title('Error')
# plt.xlabel('Steps')
# plt.ylabel('Error')
#
# plt.plot(errors, steps)
# plt.xscale("log")
# plt.yscale("log")
# plt.savefig('error_plot.png')
# plt.show()
#
#
# print(errors)
# print(steps)
