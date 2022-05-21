from export_utils import export_results
from project import FD_solver, true_sol
from utils import get_linear_index, convert_to_3D
import numpy as np
import matplotlib.pyplot as plt


def error(Nx, Ny, Nz):
    hx = 1 / (Nx - 1)
    hy = 1 / (Ny - 1)
    hz = 1 / (Nz - 1)

    u = FD_solver(Nx, Ny, Nz)
    true_solution = np.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                linear_index = get_linear_index(i, j, k, Nx, Ny)
                true_solution[linear_index] = true_sol(i * hx, j * hy, k * hz)

    # for i in range(len(u)):
    #     print(u[i], true_solution[i])
    error = np.linalg.norm(u - true_solution, ord=np.inf)
    # error = np.max(u - true_solution)
    print(f"Error: {error}")
    return error, true_solution, u

errors = []
for step in range(5, 20):
    Nx = Ny = Nz = step
    er, ts, u = error(Nx, Ny, Nz)
    ts_tensor = convert_to_3D(ts, Nx, Ny, Nz)
    u_tensor = convert_to_3D(u, Nx, Ny, Nz)
    # print(np.mean(ts_tensor), np.mean(u_tensor))
    # print(er)
    # export_results("fd-{0}.vti".format(20), u_tensor)
    # export_results("ts-{0}.vti".format(20), ts_tensor)
    print()
    errors.append(er)

plt.plot(errors)
plt.xscale("log")
plt.yscale("log")
plt.show()
