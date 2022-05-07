from math import exp, sin
import numpy as np

# the true solution u
def true_solution(x, y, z):
    return exp(x) * sin(y) + exp(y) * sin(z) + exp(z) * sin(x)

def FD_solver(h=0.1, k=0.1, l=0.1):
    # defining the grid
    x = np.arange(0, 1, h)
    y = np.arange(0, 1, k)
    z = np.arange(0, 1, l)
    
    FD_operator = (1/(h^2)) * ()