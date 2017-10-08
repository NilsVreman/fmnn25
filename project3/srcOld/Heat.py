import numpy as np
import scipy.linalg as spl

"""
length_x, length_y: room dimensions in length units
dx: grid point distance
"""
def _create_sys_matrix(length_x, length_y, dx):
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
length_x, length_y: room dimensions in length units
v1: Vector containing boundary values along left wall
v2: Vector containing boundary values along top wall
v3: Vector containing boundary values along right wall
v4: Vector containing boundary values along bottom wall
dx: Grid point distance
"""
def _create_boundary_vec(length_x, length_y, v1, v2, v3, v4, dx):
        # calculations
    nbr_sq_x = int(length_x / dx)
    nbr_sq_y = int(length_y / dx)
        # nbr = nbr of internal grid points
    nbr = nbr_sq_x - 1

        # Create boundary condition vector
    b = np.zeros((nbr_sq_x - 1) * (nbr_sq_y - 1))

        # Adds boundary conditions from LEFT boundary
    for i in range(0, len(v1) - 2):
        j = nbr * i
        b[j] = b[j] - v1[i + 1] / dx ** 2

        # Adds boundary conditions from RIGHT boundary
    for i in range(0, len(v3) - 2):
        j = (i + 1) * nbr - 1
        b[j] = b[j] - v3[i + 1] / dx ** 2

        # Adds boundary conditions from TOP boundary
    for i in range(1, len(v2) - 1):
        b[i-1] = b[i-1] - v2[i] / dx ** 2

        # Adds boundary conditions from BOTTOM boundary

        for i in range(1, len(v4) - 1):
            j = len(b) - nbr - 1 + i
            b[j] = b[j] - v4[i] / dx ** 2

        return b

    """
    Calculates the room temperature according to the laplace equation
    """
    def calculate_heat(self, length_x, length_y, v1, v2, v3, v4, dx):
        A = self._create_sys_matrix(length_x, length_y, dx)
        b = self._create_boundary_vec(length_x, length_y, v1, v2, v3, v4, dx)

        u = spl.solve(A, b)
        return u

if __name__ == '__main__':
    # input
    length_x = 1
    length_y = 2
    dx = 1 / 3
    w = 0.75

    nbr_sq_x = int(length_x / dx)
    nbr_sq_y = int(length_y / dx)

    h = Heat()
    v1 = 15 * np.ones(nbr_sq_y+1) #left
    v2 = 40 * np.ones(nbr_sq_x+1) #top
    v3 = 15 * np.ones(nbr_sq_y+1) #right
    v4 = 5  * np.ones(nbr_sq_x+1) #bottom

    t2 = 15 * np.ones(nbr_sq_x+1) #top
    t3 = 40 * np.ones(nbr_sq_x+1) #right
    t4 = 15 * np.ones(nbr_sq_x+1) #bottom 

    u_old_big = h.calculate_heat(length_x, length_y, v1, v2, v3, v4, dx)

    for i in range(0, 10):

        u = h.calculate_heat(length_x, length_y, v1, v2, v3, v4, dx)
        u = w*u + (1-w)*u_old_big
        u_old_big = u
        print(u, '\n--------------------------')
        
        neumann = -1*(v3[1:int(1/dx)]-[u[i] for i in np.arange(1, 4, 2)])
        neumann = np.append(np.append([0], neumann / dx), [0])

        t1 = neumann
        u = h.calculate_heat(1, 1, t1, t2, t3, t4, dx)
        print(u, '\n--------------------------')

        v3[1:3] = np.array([u[1], u[3]]) 
