import numpy as np
from project import f

from project import get_linear_index


def apply_dirichlet_boundary_condition_x(Nx, Ny, Nz, end=True):
    steps = [float(1) / (Nx - 1), float(1) / (Ny - 1), float(1) / (Nz - 1)]
    dims = [Nx, Ny, Nz]
    matrix = np.zeros((Nx * Ny * Nz, Nx * Ny * Nz))
    index = dims[0] - 1 if end else 0
    i = 0
    for j in range(dims[1]):
        for k in range(dims[2]):
            matrix[get_linear_index(index, j, k, dims[0], dims[1])] = f(i * steps[0], j * steps[1], k * steps[2])
    return matrix


def apply_dirichlet_boundary_condition_y(Nx, Ny, Nz, end=True):
    steps = [float(1) / (Nx - 1), float(1) / (Ny - 1), float(1) / (Nz - 1)]
    dims = [Nx, Ny, Nz]
    matrix = np.zeros((Nx * Ny * Nz, Nx * Ny * Nz))
    index = dims[1] - 1 if end else 0
    j = 0
    for i in range(dims[0]):
        for k in range(dims[2]):
            matrix[get_linear_index(index, j, k, dims[0], dims[1])] = f(i * steps[0], j * steps[1], k * steps[2])
    return matrix


def apply_dirichlet_boundary_condition_z(Nx, Ny, Nz, end=True):
    steps = [float(1) / (Nx - 1), float(1) / (Ny - 1), float(1) / (Nz - 1)]
    dims = [Nx, Ny, Nz]
    matrix = np.zeros((Nx * Ny * Nz, Nx * Ny * Nz))
    index = dims[2] - 1 if end else 0
    k = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            matrix[get_linear_index(index, j, k, dims[0], dims[1])] = f(i * steps[0], j * steps[1], k * steps[2])
    return matrix
