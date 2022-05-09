import numpy as np
from project import f, u

from project import get_linear_index


def g(x, y, z):
    return u(x, y, z)


def R_g(Nx, Ny, Nz):
    return R_x(Nx, Ny, Nz, True) + \
           R_x(Nx, Ny, Nz, False) + \
           R_y(Nx, Ny, Nz, True) + \
           R_y(Nx, Ny, Nz, False) + \
           R_z(Nx, Ny, Nz, True) + \
           R_z(Nx, Ny, Nz, False)


def R_x(Nx, Ny, Nz, end=True):
    steps = [float(1) / (Nx - 1), float(1) / (Ny - 1), float(1) / (Nz - 1)]
    dims = [Nx, Ny, Nz]
    matrix = np.zeros((Nx * Ny * Nz))
    index = dims[0] - 1 if end else 0
    i = 0
    for j in range(dims[1]):
        for k in range(dims[2]):
            matrix[get_linear_index(index, j, k, dims[0], dims[1])] = g(i * steps[0], j * steps[1], k * steps[2])
    return matrix


def R_y(Nx, Ny, Nz, end=True):
    steps = [float(1) / (Nx - 1), float(1) / (Ny - 1), float(1) / (Nz - 1)]
    dims = [Nx, Ny, Nz]
    matrix = np.zeros((Nx * Ny * Nz))
    index = dims[1] - 1 if end else 0
    j = 0
    for i in range(dims[0]):
        for k in range(dims[2]):
            matrix[get_linear_index(index, j, k, dims[0], dims[1])] = g(i * steps[0], j * steps[1], k * steps[2])
    return matrix


def R_z(Nx, Ny, Nz, end=True):
    steps = [float(1) / (Nx - 1), float(1) / (Ny - 1), float(1) / (Nz - 1)]
    dims = [Nx, Ny, Nz]
    matrix = np.zeros((Nx * Ny * Nz))
    index = dims[2] - 1 if end else 0
    k = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            matrix[get_linear_index(index, j, k, dims[0], dims[1])] = g(i * steps[0], j * steps[1], k * steps[2])
    return matrix
