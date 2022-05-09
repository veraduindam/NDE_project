from math import exp, sin
import numpy as np


# the true solution u
def u(x, y, z):
    return exp(x) * sin(2 * y) + exp(y) * sin(float(z) / 2) + exp(z) * sin(2 * x)


def f(x, y, z):
    return 3 * exp(z) * sin(2 * x) + 3 * exp(x) * sin(2 * y) - float(3) / 4 * exp(y) * sin(float(z) / 2)


def get_linear_index(i, j, k, N, M):
    return (N * M) * k + N * j + i


# function to define the tridiagonal matrices Dx, Dy and Dz of dimension N
def tridiag(N, offset):
    central = np.diag([-2] * N, 0)
    l = np.diag([1] * (N - offset), -offset)
    u = np.diag([1] * (N - offset), offset)
    # print(central.shape, l.shape, u.shape)
    return central + l + u


# define the matrix and r.h.s. vector and solve
def FD_solver(Nx=2, Ny=3, Nz=2):
    # defining the grid
    hx = 1 / (Nx - 1)
    hy = 1 / (Ny - 1)
    hz = 1 / (Nz - 1)

    # define the grid points in [0,1]^3
    x = np.linspace(0, 1, Nx, endpoint=True)
    y = np.linspace(0, 1, Ny, endpoint=True)
    z = np.linspace(0, 1, Nz, endpoint=True)

    # FD in the x direction
    Dx = tridiag(Ny * Nz * Nx, 1)
    Dy = tridiag(Nx * Nz * Ny, Nx)
    Dz = tridiag(Ny * Nx * Nz, Nx * Ny)
    print(Dx)
    print(Dy)
    print(Dz)

    print()

    Ix = np.identity(Nx)
    Iy = np.identity(Ny)
    Iz = np.identity(Nz)

    # K1 = np.kron(Iy, Dx)
    # # print('----------')
    # # print(K1)
    # K2 = np.kron(Dy, Ix)
    # K3 = np.kron(Dz, Ix)
    # print('----------')
    print(Dx + Dy + Dz)


FD_solver()
