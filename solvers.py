import numpy as np
import time
from pprint import pprint


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{0} {1} ms'.format(method.__name__, (te - ts) * 1000))
        return result

    return timed


@timeit
def jacobi_solver(A, b, num_iterations, eps=0.0001):
    x = np.zeros_like(b)
    D = np.diagflat(np.diagonal(A))
    LU = -(np.copy(A) - D)
    T = np.linalg.inv(D) @ LU
    c = np.linalg.inv(D) @ b

    for i in range(num_iterations):
        x_new = T @ x + c
        error = np.linalg.norm(x_new - x)
        x = x_new
        if error < eps:
            return x
    return x


@timeit
def gauss_siedel(A, b, num_iterations, eps=1e-4):
    x = np.zeros_like(b)
    D = np.diagflat(np.diagonal(A))
    L = -np.tril(A - D)
    U = -np.triu(A - D)
    T = np.linalg.inv(D - L) @ U
    c = np.linalg.inv(D - L) @ b

    for i in range(num_iterations):
        x_new = T @ x + c
        error = np.linalg.norm(x_new - x)
        x = x_new
        if error < eps:
            return x
    return x


A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
b = np.array([6., 25., -11., 15.])

sol = jacobi_solver(A, b, num_iterations=25)
pprint(A)
pprint(b)
pprint(sol)
pprint(A @ sol)

sol = gauss_siedel(A, b, num_iterations=25)
pprint(A)
pprint(b)
pprint(sol)
pprint(A @ sol)
