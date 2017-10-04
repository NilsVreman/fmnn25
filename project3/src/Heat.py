import numpy as np
import scipy.linalg as spl

class Heat():

    def _create_sys_matrix(self, length_x, length_y, dx):
        # calculations
        nbr_sq_x = int(length_x / dx)
        nbr_sq_y = int(length_y / dx)
        # nbr = nbr of internal grid points
        nbr = nbr_sq_x - 1

        # Create Toeplitz matrix disregarding boundary conditions
        a = np.zeros((1, (nbr_sq_x - 1) * (nbr_sq_y - 1)))
        a[0, 0] = -4.0
        a[0, 1] = a[0, nbr] = 1.0
        A = spl.toeplitz(a)

        # Considering boundary conditions and changing those values to zero
        # this is done since they will be moved to rhs
        for i in np.arange(nbr - 1, len(A) - 1, nbr):
            row = i
            A[row, row + 1] = 0
            A[row + 1, row] = 0

        A = A * 1 / dx ** 2
        return A


    """
    v1: Vector containing boundary values along left wall
    v2: Vector containing boundary values along top wall
    v3: Vector containing boundary values along right wall
    v4: Vector containing boundary values along bottom wall
    """


    def _create_boundary_vec(self, length_x, length_y, v1, v2, v3, v4, dx):
        # calculations
        nbr_sq_x = int(length_x / dx)
        nbr_sq_y = int(length_y / dx)
        # nbr = nbr of internal grid points
        nbr = nbr_sq_x - 1

        # Create boundary condition vector
        b = np.zeros((nbr_sq_x - 1) * (nbr_sq_y - 1))

        for i in range(0, len(v1) - 1):
            j = nbr * i
            b[j] = b[j] - v1[i + 1] / dx ** 2

        for i in range(0, len(v3) - 1):
            j = (i + 1) * nbr - 1
            b[j] = b[j] - v2[i + 1] / dx ** 2

        for i in range(1, len(v2) - 1):
            b[i] = b[i] - v2[i] / dx ** 2

        for i in range(1, len(v4) - 1):
            b[len(b) - nbr - 1 + i] = b[len(b) - nbr - 1 + i] - v4[i] / dx ** 2

        return b

    def calculate_heat(self, length_x, length_y, v1, v2, v3, v4, dx):
        A = self._create_sys_matrix(length_x, length_y, dx)
        b = self._create_boundary_vec(length_x, length_y, v1, v2, v3, v4, dx)

        print(b)



if __name__ == '__main__':
    # input
    length_x = 1
    length_y = 2
    dx = 1 / 3

    nbr_sq_x = int(length_x / dx)
    nbr_sq_y = int(length_y / dx)
    v1 = -10 * np.ones(nbr_sq_y+1)
    v2 = -15 * np.ones(nbr_sq_x+1)
    v3 = -20 * np.ones(nbr_sq_y+1)
    v4 = -25 * np.ones(nbr_sq_x+1)

    h = Heat()
    h.calculate_heat(length_x, length_y, v1, v2, v3, v4, dx)