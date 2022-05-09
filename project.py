from math import exp, sin
import numpy as np
from solvers import conj_grad
import itertools


# Nx, Ny, Nz - total number of nodes
def get_free_nodes_list(Nx, Ny, Nz):
    x = [i for i in range(1, Nx - 1)]
    y = [j for j in range(1, Ny - 1)]
    z = [k for k in range(1, Nz - 1)]

    free_nodes = []
    product = itertools.product(*[x, y, z])
    for element in product:
        print(element)
        free_nodes.append(get_linear_index(*element, Nx, Ny))
    return free_nodes


def F(Nx, Ny, Nz):
    steps = [float(1) / (Nx - 1), float(1) / (Ny - 1), float(1) / (Nz - 1)]
    F = np.zeros((Nx * Ny * Nz))
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                F[get_linear_index(i, j, k, Nx, Ny)] = f(i * steps[0], j * steps[1], k * steps[2])
    return F


# the true solution u
def u(x, y, z):
    return exp(x) * sin(2 * y) + exp(y) * sin(float(z) / 2) + exp(z) * sin(2 * x)


# the function f
def f(x, y, z):
    return 3 * exp(z) * sin(2 * x) + 3 * exp(x) * sin(2 * y) - float(3) / 4 * exp(y) * sin(float(z) / 2)


# get the index in the array of 
def get_linear_index(i, j, k, N, M):
    return (N * M) * k + N * j + i


# This gives us the matrices Dx, Dy and Dz
def D(N):
    return np.diag([1] * (N - 1), -1) + np.diag([-2] * N, 0) + np.diag([1] * (N - 1), 1)


def Z(Nx, Ny, Nz):
    hx = 1 / (Nx - 1)
    hy = 1 / (Ny - 1)
    hz = 1 / (Nz - 1)

    F = np.zeros(Nx * Ny * Nz)
    R_g = np.zeros(Nx * Ny * Nz)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                index = get_linear_index(i, j, k, Nx, Ny)
                F[index] = f(i * hx, j * hy, k * hz)

                if i == 0 or j == 0 or k == 0 or i == Nx - 1 or j == Ny - 1 or k == Nz - 1:
                    R_g[index] = u(i * hx, j * hy, k * hz)

    return F - R_g


Z(Nx=2, Ny=2, Nz=2)


def alt_FD_solver(Nx=4, Ny=4, Nz=4):
    # defining the grid
    hx = 1 / (Nx - 1)
    hy = 1 / (Ny - 1)
    hz = 1 / (Nz - 1)

    # define the grid points in [0,1]^3
    x = np.linspace(0, 1, Nx, endpoint=True)
    y = np.linspace(0, 1, Ny, endpoint=True)
    z = np.linspace(0, 1, Nz, endpoint=True)

    # define Dx, Dy, Dz
    Dx = D(Nx)
    Dy = D(Ny)
    Dz = D(Nz)

    # define correctly sized identity matrices
    Ix = np.identity(Nx)
    Iy = np.identity(Ny)
    Iz = np.identity(Nz)

    # define matrices K1, K2, K3
    Kx = 1 / (hx ** 2) * np.kron(np.kron(Dx, Iy), Iz)
    Ky = 1 / (hy ** 2) * np.kron(Iz, np.kron(Ix, Dy))
    Kz = 1 / (hz ** 2) * np.kron(np.kron(Ix, Dz), Iy)

    # define matrix A
    A = Kx + Ky + Kz
    z = Z(Nx, Ny, Nz)

    print(A[2][3])

    fd = get_free_nodes_list(Nx, Ny, Nz)
    product = itertools.product(*[fd, fd])
    d1 = []
    d2 = []
    for element in product:
        d1.append(element[0])
        d2.append(element[1])
    # get free nodes
    A_free = np.zeros(((Nx - 2) * (Ny - 2) * (Nz - 2), (Nx - 2) * (Ny - 2) * (Nz - 2)))
    for i in range((Nx - 2) * (Ny - 2) * (Nz - 2)):
        for j in range((Nx - 2) * (Ny - 2) * (Nz - 2)):
            A_free[i, j] = A[fd[i], fd[j]]

    print(A_free)
    print(A_free.shape)
    z_free = z[fd]
    print(z_free.shape)

    v_free = conj_grad(A_free, z_free, 25)
    v = np.zeros((Nx * Ny * Nz))
    for i in range((Nx - 2) * (Ny - 2) * (Nz - 2)):
        v[fd[i]] = v_free[i]
    print(v_free)
    # print(A)


def twoD_FD_solver(Nx=2, Ny=2):
    hx = 1 / (Nx - 1)
    hy = 1 / (Ny - 1)

    # define the grid points in [0,1]^3
    x = np.linspace(0, 1, Nx, endpoint=True)
    y = np.linspace(0, 1, Ny, endpoint=True)

    # define Dx, Dy, Dz
    Dx = D(Nx)
    Dy = D(Ny)

    # define correctly sized identity matrices
    Ix = np.identity(Nx)
    Iy = np.identity(Ny)

    K1 = (1 / hx ** 2) * np.kron(Iy, Dx)
    K2 = (1 / hy ** 2) * np.kron(Dy, Ix)

    A = K1 + K2
    print(A)


alt_FD_solver()
# fd = get_free_nodes_list(4, 4, 4)
# print(fd)
# product = itertools.product(*[fd, fd])
# d1 = []
# d2 = []
# for element in product:
#     d1.append(element[0])
#     d2.append(element[1])
#
# print(d1, d2)
