import numpy as np
import scipy.linalg as spl


def _create_sys_matrix(self, length_x, length_y, dx):

    #calculations
    nbr_sq_x = int(length_x/dx)
    nbr_sq_y = int(length_y/dx)
    #nbr = nbr of internal grid points
    nbr = nbr_sq_x - 1

    #Create Toeplitz matrix disregarding boundary conditions
    a = np.zeros((1, (nbr_sq_x-1)*(nbr_sq_y-1)))
    a[0, 0] = -4.0
    a[0, 1] = a[0, nbr] = 1.0
    A = spl.toeplitz(a)

    #Considering boundary conditions and changing those values to zero
    #this is done since they will be moved to rhs
    for i in np.arange(nbr-1, len(A)-1, nbr):
        row = i
        A[row, row+1] = 0
        A[row+1, row] = 0

    A = A*1/dx**2
    return A

if __name__ == '__main__':

    #input
    length_x = 1
    length_y = 2
    dx = 1/3
    

