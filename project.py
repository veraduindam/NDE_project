from decimal import DivisionByZero
from math import exp, sin
import numpy as np
from solvers import conj_grad, jacobi_solver
import itertools
from export_utils import export_results


# Nx, Ny, Nz are the total number of nodes
# returns list of the nodes that are not on the boundary
def get_free_nodes_list(Nx, Ny, Nz):
    x = [i for i in range(1, Nx - 1)]
    y = [j for j in range(1, Ny - 1)]
    z = [k for k in range(1, Nz - 1)]

    free_nodes = []
    product = itertools.product(*[x, y, z])
    for element in product:
        free_nodes.append(get_linear_index(*element, Nx, Ny))
    return free_nodes


# get the index in the array of (i, j, k)
def get_linear_index(i, j, k, N, M):
    return (N * M) * k + N * j + i


# go from the linear index back to the index (i, j, k)
def get_3D_index(linear_index, N, M):
    i = linear_index % N
    k = linear_index // (N * M)
    j = (linear_index - i - N * M * k) / N
    return i, int(j), k


# the true solution u
def true_sol(x, y, z):
    return exp(x) * sin(2 * y) + exp(y) * sin(float(z) / 2) + exp(z) * sin(2 * x)


# the function f
def f(x, y, z):
    return 3 * exp(z) * sin(2 * x) + 3 * exp(x) * sin(2 * y) - float(3) / 4 * exp(y) * sin(float(z) / 2)


# this gives us the matrices Dx, Dy and Dz
def D(N):
    return np.diag([1] * (N - 1), -1) + np.diag([-2] * N, 0) + np.diag([1] * (N - 1), 1)


# computation of R_g and the discrete vector F
def R_g_and_F(Nx, Ny, Nz):
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
                    R_g[index] = true_sol(i * hx, j * hy, k * hz)

    return R_g, F


# solving the problem
def FD_solver(Nx, Ny, Nz):
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
    Ky = 1 / (hy ** 2) * np.kron(Ix, np.kron(Iz, Dy))
    Kz = 1 / (hz ** 2) * np.kron(np.kron(Ix, Dz), Iy)

    # define matrix A
    A = Kx + Ky + Kz

    # define vectors R_g, F and z
    R_g, F = R_g_and_F(Nx, Ny, Nz)
    z = F - A @ R_g

    # take the part of the system that is still free and solve
    fd = get_free_nodes_list(Nx, Ny, Nz)
    A_free = A[:, fd][fd, :]
    z_free = z[fd]
    v_free = jacobi_solver(A_free, z_free, 25)

    # create v in correct dimension
    v = np.zeros((Nx * Ny * Nz))
    for i in range((Nx - 2) * (Ny - 2) * (Nz - 2)):
        v[fd[i]] = v_free[i]

    # define solution u
    u = R_g + v

    return u


# function needed for plotting the solution: 
def convert_to_3D(u, Nx, Ny, Nz):
    tensor = np.zeros((Nx, Ny, Nz))
    for l in range(len(u)):
        index = get_3D_index(l, Nx, Ny)
        tensor[index[0], index[1], index[2]] = u[l]
    return tensor
        

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

u = FD_solver(4, 4, 4)
u = convert_to_3D(u, 4, 4, 4)
export_results("FD_solution.vti", u)
