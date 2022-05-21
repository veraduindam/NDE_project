import numpy as np

# get the index in the array of (i, j, k)
def get_linear_index(i, j, k, N, M):
    return (N * M) * k + N * j + i


# go from the linear index back to the index (i, j, k)
def get_3D_index(linear_index, N, M):
    i = linear_index % N
    k = linear_index // (N * M)
    j = (linear_index - i - N * M * k) / N
    return i, int(j), k

# function needed for plotting the solution:
def convert_to_3D(u, Nx, Ny, Nz):
    tensor = np.zeros((Nx, Ny, Nz))
    for l in range(len(u)):
        index = get_3D_index(l, Nx, Ny)
        tensor[index[0], index[1], index[2]] = u[l]
    return tensor
