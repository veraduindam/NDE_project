from math import exp, sin
import numpy as np

# the true solution u
def u(x, y, z):
    return exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x)

# function to define the tridiagonal matrices Dx, Dy and Dz of dimension N
def tridiag(N):
    return np.diag([1] * (N - 1), -1) + np.diag([-2] * N, 0) + np.diag([1] * (N - 1), 1)

# define the matrix and r.h.s. vector and solve
def FD_solver(Nx=3, Ny=2, Nz=5):
    # defining the grid
    hx = 1/(Nx - 1)
    hy = 1/(Ny - 1)
    hz = 1/(Nz - 1)

    # define the grid points in [0,1]^3
    x = np.linspace(0, 1, Nx, endpoint=True)
    y = np.linspace(0, 1, Ny, endpoint=True)
    z = np.linspace(0, 1, Nz, endpoint=True)

    # FD in the x direction
    Dx = tridiag(Nx)
    Iyz = np.identity(Ny * Nz)
    K1 = np.kron(Iyz, Dx)

    # FD in the y direction
    Dy = tridiag(Ny)
    Ixz = np.identity(Nx * Nz)
    K2 = np.kron(Dy, Ixz)

    # FD in the z direction
    Dz = tridiag(Nz)
    Ixy = np.identity(Nx * Ny)
    K3 = np.kron(Ixy, Dz)

    # the FD matrix
    A = - (1/(hx**2) * K1 + 1/(hy**2) * K2 + 1/(hz**2) * K3)

    F = np.array([0] * (Nx * Ny * Nz))

    U = np.linalg.solve(A, F)

    print(U)

FD_solver()