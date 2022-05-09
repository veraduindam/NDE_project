from math import exp, sin
import numpy as np


# the true solution u
def u(x, y, z):
    return exp(x) * sin(2 * y) + exp(y) * sin(float(z) / 2) + exp(z) * sin(2 * x)


# the function f
def f(x, y, z):
    return 3 * exp(z) * sin(2 * x) + 3 * exp(x) * sin(2 * y) - float(3) / 4 * exp(y) * sin(float(z) / 2)


# get the index in the array of 
def get_linear_index(i, j, k, N, M):
    return (N * M) * k + N * j + i


def D(N):
    return np.diag([1]* (N - 1), -1) + np.diag([-2] * N, 0) + np.diag([1] * (N - 1), 1)
    

def alt_FD_solver(Nx=3, Ny=3, Nz=3):
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

    # define matrix A
    A = Kx + Ky + Kz

    F = np.zeros(Nx * Ny * Nz)
    # define vector F
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                F[get_linear_index(i, j, k, Nx, Ny)] = f(i * hx, j * hy, k * hz)

    print(A)


def twoD_FD_solver(Nx=2, Ny=2):
    hx = 1/(Nx - 1)
    hy = 1/(Ny - 1)

    # define the grid points in [0,1]^3
    x = np.linspace(0, 1, Nx, endpoint=True)
    y = np.linspace(0, 1, Ny, endpoint=True)

    # define Dx, Dy, Dz
    Dx = D(Nx)
    Dy = D(Ny)

    # define correctly sized identity matrices
    Ix = np.identity(Nx)
    Iy = np.identity(Ny)

    K1 = (1/hx**2) * np.kron(Iy, Dx)
    K2 = (1/hy**2) * np.kron(Dy, Ix)

    A = K1 + K2
    print(A)

twoD_FD_solver()
alt_FD_solver()
