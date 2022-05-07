from math import exp, sin
import numpy as np

# the true solution u
def u(x, y, z):
    return exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x)

# function to define the tridiagonal matrices Dx, Dy and Dz of dimension N
def tridiag(N):
    return np.diag([1] * (N - 1), -1) + np.diag([-2] * N, 0) + np.diag([1] * (N - 1), 1)

# define the matrix and r.h.s. vector and solve
def FD_solver(Nx=2, Ny=2, Nz=2):
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
    Dy = tridiag(Ny)
    Dz = tridiag(Nz)

    Ix = np.identity(Nx)
    Iy = np.identity(Ny)
    Iz = np.identity(Nz)

    K1 = np.kron(Iy, Dx)
    print(K1)
    K2 = np.kron(Dy, Ix)
    print(K1 + K2)


FD_solver()