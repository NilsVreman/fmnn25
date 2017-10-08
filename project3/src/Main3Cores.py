from Room_temp import Room_temp
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    '''
    Rank 0 = big room (B_rt)
    Rank 1 = small room left (S_rt)
    Rank 2 = small room right (S_rt)
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != 3:
        raise Exception("This program needs to be run by 3 processors")
    length_B_x = 1
    length_B_y = 2
    length_S_x = 1
    length_S_y = 1
    dx = 1/20
    nbr_iter = 10
    w = 0.75

    if rank == 0:
        room = Room_temp(length_B_x, length_B_y, dx)
        # temperatures corresponding to each wall for big room
        initial_wall_temp = [15, 15, 5, 40] #left, right, bottom, top
    else:
        room = Room_temp(length_S_x, length_S_y, dx)
        # temperatures corresponding to each wall for small rooms
        initial_wall_temp = [40, 40, 15, 15] #left, right, bottom, top

    # Creating boundary condition grid vectors / AKA Walls
    walls = np.array(room.create_walls())
    values = [
        np.ones(room.nbr_pts_y)* initial_wall_temp[0], #left
        np.ones(room.nbr_pts_y)* initial_wall_temp[1], #right
        np.ones(room.nbr_pts_x)* initial_wall_temp[2], #bottom
        np.ones(room.nbr_pts_x)* initial_wall_temp[3] #top
    ]
    #Add fixed conditions on unshared walls and initial guess on shared walls for small and big room
    # the conditions for small rooms on shared walls  doesn't matter since they will be overridden
    for i in range(len(walls)):
        room.add_dirichlet_cond(walls[i], values[i])
    if rank == 0:
        B_rt = room
        B_walls = walls
    else:
        S_rt = room
        S_walls = walls

    #Neumann-Dirichlet Iteration
    for i in range(0, nbr_iter):
        if rank == 0:
            #STEP 1 - calculate temp in room 2
            u_B = B_rt()
            B_rt_mat = B_rt.find_temp_matrix(u_B)
            #Calculate neumann cond for room 1 and 3
            gamma1 = 1*(B_rt_mat[int(len(B_rt_mat)/2):, 0] - B_rt_mat[int(len(B_rt_mat)/2):, 1])
            gamma2 = 1*(B_rt_mat[:int(len(B_rt_mat) / 2 + 1), -1] - B_rt_mat[:int(len(B_rt_mat) / 2 + 1), -2])

            #Send gamma to other cores -> Step 2
            comm.send(gamma1, dest=1, tag=i)
            comm.send(gamma2, dest=2, tag=i)
            u_B_new = B_rt()
            #STEP 3 - relaxation and then adding dirichlet cond to room 2
            u_B = u_B_new if i == 0 else w * u_B_new + (1-w) * u_B

            #Add new dirichlet temp to Big room
            SL_rt_mat = comm.recv(source=1, tag=i)
            SR_rt_mat = comm.recv(source=2, tag=i)

            B_rt.add_dirichlet_cond(B_walls[0][int(len(B_walls[0]) / 2):, :], SL_rt_mat)
            B_rt.add_dirichlet_cond(B_walls[1][:int(len(B_walls[1]) / 2 + 1), :], SR_rt_mat)
        else:
            #STEP 2 - calculate the temp in room 1 and 3
            gamma = comm.recv(source=0, tag=i)
            #Different walls to add neumann cond to
            if rank == 1:
                S_rt.add_neumann_cond(S_walls[1], gamma) #wall towards big room (right)
            else:
                S_rt.add_neumann_cond(S_walls[0], gamma) #wall towards big room (left)
            u_S_new = S_rt()
            # relaxation
            u_S = u_S_new if i == 0 else w*u_S_new + (1-w)*u_S
            S_rt_mat = S_rt.find_temp_matrix(u_S)

            data = S_rt_mat[:, -1] if rank == 1 else S_rt_mat[:,0]
            comm.send(data, dest=0, tag=i)

    if rank == 0:
        #Assemble final apartment matrix
        SL_rt_mat = comm.recv(source=1, tag=nbr_iter+1)
        SR_rt_mat = comm.recv(source=2, tag=nbr_iter+1)
        print('Temperature in room 1 is now: \n', SR_rt_mat)
        print('Temperature in room 2 is now: \n', B_rt_mat)
        print('Temperature in room 3 is now: \n', SL_rt_mat)
        M = np.zeros((len(B_rt_mat), len(SL_rt_mat[0]) + len(B_rt_mat[0]) + len(SR_rt_mat[0]) -2))
        M[len(SL_rt_mat)-1:, :len(SL_rt_mat[0])] = SL_rt_mat[:,:]
        M[:, len(SL_rt_mat[0])-1:len(SL_rt_mat[0])-1+len(B_rt_mat[0])] = B_rt_mat[:,:]
        M[:len(SR_rt_mat), len(SL_rt_mat[0]) + len(B_rt_mat[0]) -2:] = SR_rt_mat[:,:]

        plt.imshow(M, cmap='plasma')
        plt.colorbar(orientation='vertical')
        plt.show()
    else:
        comm.send(S_rt_mat, dest=0, tag=nbr_iter+1)
