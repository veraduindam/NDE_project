from project import FD_solver, true_sol, get_linear_index
import numpy as np

def error(Nx, Ny, Nz):
    hx = 1/(Nx - 1)
    hy = 1/(Ny - 1)
    hz = 1/(Nz - 1)

    u = FD_solver(Nx, Ny, Nz)
    true_solution = np.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                linear_index = get_linear_index(i, j, k, Nx, Ny)
                true_solution[linear_index] = true_sol(i * hx, j * hy, k * hz)
    
    # for i in range(len(u)):
    #     print(u[i], true_solution[i])
    error = np.linalg.norm(u - true_solution, ord=np.inf)
    print(f"Error: {error}")
    return error

error(10, 10, 10)
