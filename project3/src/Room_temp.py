import numpy as np
import scipy.linalg as spl

class Room_temp:

    def __init__(self, length_x, length_y, dx):
        self._length_x = length_x
        self._length_y = length_y
        self._dx = dx

        #Calculate useful parameters
        nbr_pts_x = int(length_x/dx + 1)
        nbr_pts_y = int(length_y/dx + 1)
        size = nbr_pts_x*nbr_pts_y
       
        #Create a toeplitz matrix to start with
        a = np.zeros(size)
        a[0] = -4.0
        a[1] = a[nbr_pts_x] = 1.0
        A = spl.toeplitz(a)
        
        #NÄSTA STEG ÄR ATT SKAPA EN FUNKTION SOM LÄGGER TILL ETT VILLKOR
        for row in range(0, size):
            if row < nbr_pts_x or row % nbr_pts_x == 0 or row % nbr_pts_x == nbr_pts_x-1 or row > size-nbr_pts_x:
                A[row, :] = np.zeros(size)

        self.nbr_pts_x = nbr_pts_x
        self.nbr_pts_y = nbr_pts_y
        self._A = A
        self._b = np.zeros(size)

    """
    points: the pair of (row, column) index for the point to change in the u vector (zero-based)
    values: the corresponding temperature
    """
    def add_dirichlet_cond(self, points, values):
        nbr_pts_x = self.nbr_pts_x

        for i in range(0, len(points)):
            row = points[i, 0]
            col = points[i, 1]
            pos = int(row*nbr_pts_x+col)

            self._b[pos] = values[i]
            self._A[pos, pos] = 1

        """
        print(self._A)
        print(self._b)
        """

    """
    points: the pair of (row, column) index for the point to change in the u vector (zero-based)
    values: the corresponding temperature
    """
    def add_neumann_cond(self, points, values):
        nbr_pts_x = self.nbr_pts_x

        for i in range(0, len(points)):
            row = points[i, 0]
            col = points[i, 1]
            pos = int(row*nbr_pts_x+col)

            self._b[pos] = values[i]
            self._A[pos, pos] = -1
            if col == 0:
                self._A[pos, pos+1] = 1
            else:
                self._A[pos, pos-1] = 1

        """
        print(self._A)
        print(self._b)
        """

    """
    u: the calculated temperature vector
    return: u MATRIX where each point (i, j) corresponds to the temp in the corresponding room grid point (x_i, x_j)
    """
    def find_temp_matrix(self, u):
        mat = np.zeros((self.nbr_pts_y, self.nbr_pts_x))
        for i in range(0, self.nbr_pts_y):
            mat[i, :] = u[i*self.nbr_pts_x:(i+1)*self.nbr_pts_x]

        return mat

    """
    return: u VECTOR from solving the linear system A*u = b.
    """
    def __call__(self):
        return spl.solve(self._A, self._b)


    def create_walls(self):
        nbr_pts_x = self.nbr_pts_x
        nbr_pts_y = self.nbr_pts_y
 
        # Createing boundary condition grid vectors / AKA Walls
        wallL = np.zeros((nbr_pts_y, 2))
        wallR = np.zeros((nbr_pts_y, 2))
        wallB = np.zeros((nbr_pts_x, 2))
        wallT = np.zeros((nbr_pts_x, 2))
        wallL[:, 0] = np.arange(nbr_pts_y)
        wallR[:, 0] = np.arange(nbr_pts_y)
        wallR[:, 1] = nbr_pts_x - 1
        wallB[:, 0] = nbr_pts_y - 1    
        wallB[:, 1] = np.arange(nbr_pts_x)
        wallT[:, 0] = 0
        wallT[:, 1] = np.arange(nbr_pts_x)

        return wallL, wallR, wallB, wallT
