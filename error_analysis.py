from utils import get_linear_index, convert_to_3D
import numpy as np


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
