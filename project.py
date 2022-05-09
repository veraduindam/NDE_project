from math import exp, sin
import numpy as np

# the true solution u
def u(x, y, z):
    return exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x)

# function to define the tridiagonal matrices Dx, Dy and Dz of dimension N
def tridiag(N, offset):
    central = np.diag([-2] * N, 0)
    l = np.diag([1] * (N - offset), -offset)
    u = np.diag([1] * (N - offset), offset)
    # print(central.shape, l.shape, u.shape)
    return central + l + u

def D(N):
    return np.diag([1]* (N - 1), -1) + np.diag([-2] * N, 0) + np.diag([1] * (N - 1), 1)

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

    Dx = 1/(hx**2) * tridiag(Ny * Nz * Nx, 1)
    Dy = 1/(hy**2) * tridiag(Nx * Nz * Ny, Nx)
    Dz = 1/(hz**2) * tridiag(Ny * Nx * Nz, Nx * Ny)
    # print(Dx)
    # print(Dy)
    # print(Dz)

    A = Dx + Dy + Dz
    # print(A)

def alt_FD_solver(Nx=2, Ny=2, Nz=2):
    # defining the grid
    hx = 1/(Nx - 1)
    hy = 1/(Ny - 1)
    hz = 1/(Nz - 1)

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
    Kx = 1/(hx**2) * np.kron(np.kron(Dx, Iy), Iz)
    Ky = 1/(hy**2) * np.kron(Iz, np.kron(Ix, Dy))
    Kz = 1/(hz**2) * np.kron(np.kron(Ix, Dz), Iy)

    A = Kx + Ky + Kz
    print(A)

FD_solver()
alt_FD_solver()