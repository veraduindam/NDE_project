# python implementation of the theta method
from math import exp, sin, cos
import numpy as np
import math

from error_analysis import error
from project import D, get_linear_index
from solvers import jacobi_solver, conj_grad
from export_utils import export_time_dependent
import matplotlib.pyplot as plt

MAX_TIME = 0.3


def td_true_solution_tensor_time_step(func, Nx, Ny, Nz, time):
    hx = 1 / (Nx - 1)
    hy = 1 / (Ny - 1)
    hz = 1 / (Nz - 1)
    true_solution = np.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                linear_index = get_linear_index(i, j, k, Nx, Ny)
                true_solution[linear_index] = func(i * hx, j * hy, k * hz, time)
    return true_solution


# true solution u
def true_sol(x, y, z, t):
    return exp(-t * (x ** 2 + y ** 2 + z ** 2)) * (exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x))


# the laplacion of the true solution
def laplacian(x, y, z, t):
    return -2 * exp(-t * (x ** 2 + y ** 2 + z ** 2)) * t * (
            2 * exp(z) * x * cos(x) + 2 * exp(x) * y * cos(y) + 2 * exp(y) * z * cos(z) - exp(z) * (
            -3 - 2 * z + 2 * t * (x ** 2 + y ** 2 + z ** 2)) * sin(x) + 3 * exp(x) * sin(y) + 2 * exp(
        x) * x * sin(y) - 2 * exp(x) * t * x ** 2 * sin(y) - 2 * exp(x) * t * y ** 2 * sin(y) - 2 * exp(
        x) * t * z ** 2 * sin(y) + 3 * exp(y) * sin(z) - 2 * exp(y) * t * x ** 2 * sin(z) + 2 * exp(y) * y * sin(
        z) - 2 * exp(y) * t * y ** 2 * sin(z) - 2 * exp(y) * t * z ** 2 * sin(z))

# def laplacian(x, y, z, t):
#     return (2 * t * (-2 * math.e ** z * x * cos(x) - 2 * math.e ** x * y * cos(y) -
#                      -      2 * math.e ** y * z * cos(z) - 3 * math.e ** z * sin(x) +
#                      -      2 * math.e ** z * t * x ** 2 * sin(x) + 2 * math.e ** z * t * y ** 2 * sin(x) -
#                      -      2 * math.e ** z * z * sin(x) + 2 * math.e ** z * t * z ** 2 * sin(x) -
#                      -      3 * math.e ** x * sin(y) - 2 * math.e ** x * x * sin(y) +
#                      -      2 * math.e ** x * t * x ** 2 * sin(y) + 2 * math.e ** x * t * y ** 2 * sin(y) +
#                      -      2 * math.e ** x * t * z ** 2 * sin(y) - 3 * math.e ** y * sin(z) +
#                      -      2 * math.e ** y * t * x ** 2 * sin(z) - 2 * math.e ** y * y * sin(z) +
#                      -      2 * math.e ** y * t * y ** 2 * sin(z) + 2 * math.e ** y * t * z ** 2 * sin(
#                 z))) - math.e ** (t * (x ** 2 + y ** 2 + z ** 2))


# the first order derivative in the t direction of the true solution
def u_t(x, y, z, t):
    return -(x ** 2 + y ** 2 + z ** 2) * exp(-t * (x ** 2 + y ** 2 + z ** 2)) * (
            exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x))


# the rhs function f
def f(x, y, z, t, c):
    return u_t(x, y, z, t) - c * laplacian(x, y, z, t)


# the initial condition u(x, y, z, 0)
def initial_condition(x, y, z):
    return exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x)


def td_true_solution_tensors(func, Nx, Ny, Nz, n_time_steps):
    times = np.linspace(0, MAX_TIME, n_time_steps + 1, endpoint=True)
    u_true = [td_true_solution_tensor_time_step(func, Nx, Ny, Nz, t) for t in times]
    return u_true


# implementation of the theta method
def theta_method(theta, c, N, M):
    # defining the space and time grid
    h = 1 / (N - 1)
    l = h

    lamb = c * l / (h ** 2)

    # defining the initial vector U0
    U0 = np.zeros(N ** 3)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                index = get_linear_index(i, j, k, N, N)
                U0[index] = initial_condition(i * h, j * h, k * h)

    # check the stability condition
    if theta < 0.5:
        l = h ** 2 / (2 * (1 - 2 * theta))
        # l_stab = 0.9 * h ** 2 / (2 * c * (1 - 2 * theta))
        # if l > l_stab:
        #     M = round(MAX_TIMmath.e / l_stab)
        #     l = MAX_TIMmath.e / M

    # define matrix T using kronecker products
    D_mat = D(N)
    I = np.identity(N)
    K1 = np.kron(np.kron(D_mat, I), I)
    K2 = np.kron(I, np.kron(I, D_mat))
    K3 = np.kron(np.kron(I, D_mat), I)
    T = K1 + K2 + K3

    id = np.identity(N ** 3)
    A = id - lamb * theta * T
    B = id + (1 - theta) * lamb * T

    U_theta = [U0]
    U_previous = U0
    F_old = np.zeros(N ** 3)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                index = get_linear_index(i, j, k, N, N)
                F_old[index] = f(i * h, j * h, k * h, 0, c)

    for m in range(1, M):
        t = l * m

        # boundary condition vector Z and vector F at that point in time
        F = np.zeros(N ** 3)
        Z = np.zeros(N ** 3)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    index = get_linear_index(i, j, k, N, N)
                    F[index] = theta * lamb * f(i * h, j * h, k * h, (l + 1) * m, c) + (1 - theta) * lamb * f(i * h,
                                                                                                              j * h,
                                                                                                              k * h,
                                                                                                              l * m, c)

                    if i == 0 or j == 0 or k == 0 or i == N - 1 or j == N - 1 or k == N - 1:
                        Z[index] = theta * lamb * true_sol(i * h, j * h, k * h, (l + 1) * m) + (
                                1 - theta) * lamb * true_sol(i * h, j * h, k * h, l * m)

        Y = B @ U_previous + Z + F
        U = conj_grad(A, Y, 200)

        U_theta.append(U)
        U_previous = U
        # F_old = F

    return U_theta


N = 10
M = 20
# print(f(0, 0, 0, 0, ))
# print(f(x=1, y=0, z=0, t=0, c=1))
# print(f(x=0, y=1, z=0, t=0, c=1))
# print(f(x=0, y=0, z=1, t=0, c=1))
# print(f(x=0, y=0, z=1, t=1, c=1))

# errors = []
# for m in range(5, 20):
#     l = np.pi / m
#     last_time = np.pi - l
#     ut = theta_method(0.5, 1, N, M)
#     ts = td_true_solution_tensor_time_step(true_sol, N, N, N, last_time)
#     er = error(ts, ut)
#     errors.append(er)
#
# plt.plot(errors)
# plt.xscale("log")
# plt.yscale("log")
# plt.show()
ut = theta_method(0.5, 1, N, M)
export_time_dependent(ut, 'tds', N, './td_theta_solution')
u_true = td_true_solution_tensors(true_sol, N, N, N, M)
export_time_dependent(u_true, 'tds', N, './td_true_solution')
