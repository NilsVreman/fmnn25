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
def calculate_heat(length_x, length_y, v1, v2, v3, v4, dx):
    A = _create_sys_matrix(length_x, length_y, dx)
    print(A)
    b = _create_boundary_vec(length_x, length_y, v1, v2, v3, v4, dx)
    print(b)
    u = spl.solve(A, b)
    return u


"""if __name__ == '__main__':
# input
length_x = 1
length_y = 1
dx = 1 / 4
nbr_sq_x = int(length_x / dx)
nbr_sq_y = int(length_y / dx)
v1 = 10 * np.ones(nbr_sq_y+1)
v2 = 15 * np.ones(nbr_sq_x+1)
v3 = 20 * np.ones(nbr_sq_y+1)
v4 = 25 * np.ones(nbr_sq_x+1)

h = Heat()
h.calculate_heat(length_x, length_y, v1, v2, v3, v4, dx)"""
    
dx = 1/3
length_s = int((1 / dx) + 1)
length_b_x = length_s
length_b_y = length_s * 2 - 1

# Small walls
small_tb = 15 * np.ones(length_s)
sl_left = 40 * np.ones(length_s)
sl_right = 15 * np.ones(length_s)

sr_left = 15 * np.ones(length_s)
sr_right = 40 * np.ones(length_s)

# Big walls
b_top = 40 * np.ones(length_s)
b_bottom = 5 * np.ones(length_s)
b_left = 15 * np.ones(length_b_y)
b_right = 15 * np.ones(length_b_y)


for x in range(10):
    # Big A-matrix
    big_A = _create_sys_matrix(1 , 2, dx)
    big_b = _create_boundary_vec(1, 2, b_left, b_top, b_right, b_bottom, dx)
    
    u_big_new = spl.solve(big_A, big_b)
    #print(u_big_new)
    
    if x == 0:
        u_big = u_big_new
    else:
        u_big = 0.8 * u_big + 0.2 * u_big_new
        
    print(u_big)
    print()
        
    # Calculate left
    for i in range(1, length_s - 1):
        sl_right[i] =  u_big[length_s + i * 2] - b_left[length_s]
    
    sl_A = _create_sys_matrix(1, 1, dx)
    sl_b = _create_boundary_vec(1, 1, sl_left, small_tb, sl_right, small_tb, dx)
    
    u_sl_new = spl.solve(sl_A, sl_b)
    
    if x == 0:
        u_sl = u_sl_new
    else:
        u_sl = 0.8 * u_sl + 0.2 * u_sl_new
    
    # Calculate right
    
    for i in range(1, length_s - 1):
        sr_left[i] = u_big[i * 2 - 1] - b_right[i] 
        
    sr_A = _create_sys_matrix(1, 1, dx)
    sr_b = _create_boundary_vec(1, 1, sr_left, small_tb, sr_right, small_tb, dx)
    
    u_sr_new = spl.solve(sr_A, sr_b)
    
    if x == 0:
        u_sr = u_sr_new
    else:
        u_sr = 0.8 * u_sr + 0.2 * u_sr_new 
    
    
    # Calculate b_left and b_right
    
    b_left[4] = u_sl[1]
    b_left[5] = u_sl[3]
    
    
    
    b_right[1] = u_sr[0]
    b_right[2] = u_sr[2]
    
        
        
        
        