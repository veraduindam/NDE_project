# python implementation of the theta method
from math import exp, sin, cos
import numpy as np
from project import D, get_linear_index
from solvers import jacobi_solver

# true solution u
def true_sol(x, y, z, t):
    return exp(-t * (x**2 + y**2 + z**2)) * (exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x))


def laplacian(x, y, z, t):
    return -2 * exp(-t * (x**2 + y**2 + z**2)) * t * (2 * exp(z) * x * cos(x) + 2 * exp(x) * y * cos(y) + 2 * exp(y) * z * cos(z) - exp(z) * (-3 - 2 * z + 2 * t * (x**2 + y**2 + z**2)) * sin(x) + 3 * exp(x) * sin(y) + 2 * exp(x) * x * sin(y) - 2 * exp(x) * t * x**2 * sin(y) - 2 * exp(x) * t * y**2 * sin(y) - 2 * exp(x) * t * z**2 * sin(y) + 3 * exp(y) * sin(z) - 2 * exp(y) * t * x**2 * sin(z) + 2 * exp(y) * y * sin(z) - 2 * exp(y) * t * y**2 * sin(z) - 2 * exp(y) * t * z**2 * sin(z))


def u_t(x, y, z, t):
    return -(x**2 + y**2 + z**2) * exp(-t * (x**2 + y**2 + z**2)) * (exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x))


def f(x, y, z, t, c):
    return u_t(x, y, z, t) - c * laplacian(x, y, z, t)


def initial_condition(x, y, z):
    return exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x)


def theta_method(theta, c, N, M):
    # defining the space and time grid
    h = 1 / (N - 1)
    l = np.pi / M

    lamb = c * l / (h**2)

    # defining the initial vector U0
    U0 = np.zeros(N**3)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                index = get_linear_index(i, j, k, N, N)
                U0[index] = initial_condition(i * h, j * h, k * h)

    # check the stability condition
    if theta < 0.5:
        l_stab = 0.9 * h**2 / (2 * c * (1 - 2 * theta))
        if l > l_stab:
            M = round(np.pi / l_stab)
            l = np.pi / M

    # define matrix T using kronecker products
    D_mat = D(N)
    I = np.identity(N)
    K1 = np.kron(np.kron(D_mat, I), I)
    K2 = np.kron(I, np.kron(I, D_mat))
    K3 = np.kron(np.kron(I, D_mat), I)
    T = K1 + K2 + K3

    id = np.identity(N**3)
    A = id - lamb * theta * T
    B = id + (1 - theta) * lamb * T

    U_theta = [U0]
    U_previous = U0
    for m in range(1, M):
        t = l * m 

        # boundary condition vector Z and vector F at that point in time
        F = np.zeros(N**3)
        Z = np.zeros(N**3)

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    index = get_linear_index(i, j, k, N, N)
                    F[index] = f(i * h, j * h, k * h, t, c)
                    
                if i == 0 or j == 0 or k == 0 or i == N - 1 or j == N - 1 or k == N - 1:
                    Z[index] = theta * lamb * true_sol(i * h, j * h, k * h, (l + 1) * m) + (1 - theta) * lamb * true_sol(i * h, j * h, k * h, l * m)

        Y = B @ U_previous + Z + F
        U = jacobi_solver(A, Y, 25)

        U_theta.append(U)
        U_previous = U

    print(U_theta)
    return U_theta

theta_method(0.5, 1, 2, 10)






