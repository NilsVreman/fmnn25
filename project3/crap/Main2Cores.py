from Room_temp import Room_temp
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    
    length_B_x = 1
    length_B_y = 2
    length_S_x = 1
    length_S_y = 1
    dx = 1/20
    nbr_iter = 10
    w = 0.75
 
    if rank == 0:
        B_rt = Room_temp(length_B_x, length_B_y, dx)
        SL_rt = Room_temp(length_S_x, length_S_y, dx)
    elif rank == 1:
        SR_rt = Room_temp(length_S_x, length_S_y, dx)

    # Nbr grid points in different rooms and directions
    
    if rank == 0:
        B_nbr_pts_x = B_rt.nbr_pts_x
        B_nbr_pts_y = B_rt.nbr_pts_y
        S_nbr_pts_x = SL_rt.nbr_pts_x
        S_nbr_pts_y = SL_rt.nbr_pts_y
        
        # Createing boundary condition grid vectors / AKA Walls
        B_wallL, B_wallR, B_wallB, B_wallT = B_rt.create_walls()
        S_wallL, S_wallR, S_wallB, S_wallT = SL_rt.create_walls()
    elif rank == 1:
        S_nbr_pts_x = SR_rt.nbr_pts_x
        S_nbr_pts_y = SR_rt.nbr_pts_y
        # Createing boundary condition grid vectors / AKA Walls
        S_wallL, S_wallR, S_wallB, S_wallT = SR_rt.create_walls()
    

    
    
    

    # temperatures corresponding to each wall
    if rank == 0:
        #Big
        B_valuesL = np.ones(B_nbr_pts_y)*15
        B_valuesR = np.ones(B_nbr_pts_y)*15
        B_valuesB = np.ones(B_nbr_pts_x)*5
        B_valuesT = np.ones(B_nbr_pts_x)*40
        #Small left
        SL_valuesL = np.ones(S_nbr_pts_y)*40
        SL_valuesR = np.ones(S_nbr_pts_y) #To be calculated
        SL_valuesB = np.ones(S_nbr_pts_x)*15
        SL_valuesT = np.ones(S_nbr_pts_x)*15
    elif rank == 1:
        #Small right
        SR_valuesL = np.ones(S_nbr_pts_y) #To be calculated
        SR_valuesR = np.ones(S_nbr_pts_y)*40
        SR_valuesB = np.ones(S_nbr_pts_x)*15
        SR_valuesT = np.ones(S_nbr_pts_x)*15
    
    #Add fixed conditions on unshared walls
    if rank == 0:
        #Big
        B_rt.add_dirichlet_cond(B_wallB, B_valuesB)
        B_rt.add_dirichlet_cond(B_wallT, B_valuesT)
        #Small left
        SL_rt.add_dirichlet_cond(S_wallL, SL_valuesL)
        SL_rt.add_dirichlet_cond(S_wallT, SL_valuesT)
        SL_rt.add_dirichlet_cond(S_wallB, SL_valuesB)
        #Add initial guess on shared walls for B_rt
        B_rt.add_dirichlet_cond(B_wallL, B_valuesL)
        B_rt.add_dirichlet_cond(B_wallR, B_valuesR)
    elif rank == 1:
        #Small right
        SR_rt.add_dirichlet_cond(S_wallR, SR_valuesR)
        SR_rt.add_dirichlet_cond(S_wallT, SR_valuesT)
        SR_rt.add_dirichlet_cond(S_wallB, SR_valuesB)
            
    #Neumann-Dirichlet Iteration
    for i in range(0, nbr_iter):
        
        if rank == 0:
            
            #STEP 1 - calculate temp in room 2
            u_B = B_rt()
            B_rt_mat = B_rt.find_temp_matrix(u_B)
            #Calculate neumann cond for room 1 and 3
            gamma1 = 1*(B_rt_mat[int(len(B_rt_mat)/2):, 0] - B_rt_mat[int(len(B_rt_mat)/2):, 1])
            gamma2 = 1*(B_rt_mat[:int(len(B_rt_mat) / 2 + 1), -1] - B_rt_mat[:int(len(B_rt_mat) / 2 + 1), -2])
            
            #Send gamma2 to other core
            comm.send(gamma2, dest=1, tag=i)
            print("core 0 sent msg")
            
            #Add the neumann condition on gamma1 to wall
            SL_rt.add_neumann_cond(S_wallR, gamma1)
            

            #STEP 2 - calculate the temp in room 1 and 3
            u_SL_new = SL_rt()
            u_B_new = B_rt()

            #STEP 3 - relaxation and then adding dirichlet cond to room 2
            if i == 0:
                u_SL = u_SL_new
                u_B = u_B_new
            else:
                u_B = w * u_B_new + (1-w) * u_B
                u_SL = w * u_SL_new + (1-w) * u_SL

            #Add new dirichlet temp to Big room
            SL_rt_mat = SL_rt.find_temp_matrix(u_SL)

            data = comm.recv(source=1, tag=i) 
            print("core 0 recv msg")
            
            B_rt.add_dirichlet_cond(B_wallL[int(len(B_wallL) / 2):, :], SL_rt_mat[:,-1])
            B_rt.add_dirichlet_cond(B_wallR[:int(len(B_wallR) / 2 + 1), :], data)
        
        elif rank == 1:
            
            gamma2 = comm.recv(source=0, tag=i)
            print("core 1 recv msg")
            
            SR_rt.add_neumann_cond(S_wallL, gamma2)
            u_SR_new = SR_rt()
            
            if i == 0:
                u_SR = u_SR_new
            else:
                u_SR = w*u_SR_new + (1-w)*u_SR
            
            SR_rt_mat = SR_rt.find_temp_matrix(u_SR)
            
            data = SR_rt_mat[:, 0]
            
            comm.send(data, dest=0, tag=i)
            print("core 1 send msg")
    if rank == 0:
        SR_rt_mat = comm.recv(source=1, tag=100)
    elif rank == 1:
        comm.send(SR_rt_mat, dest=0, tag=100)
        
    print('Temperature in room 1 is now: \n', SL_rt_mat)
    print('Temperature in room 2 is now: \n', B_rt_mat)
    print('Temperature in room 3 is now: \n', SR_rt_mat)
    #Assemble final apartment matrix
    M = np.zeros((len(B_rt_mat), len(SL_rt_mat[0]) + len(B_rt_mat[0]) + len(SR_rt_mat[0]) -2))
    M[len(SL_rt_mat)-1:, :len(SL_rt_mat[0])] = SL_rt_mat[:,:]
    M[:, len(SL_rt_mat[0])-1:len(SL_rt_mat[0])-1+len(B_rt_mat[0])] = B_rt_mat[:,:]
    M[:len(SR_rt_mat), len(SL_rt_mat[0]) + len(B_rt_mat[0]) -2:] = SR_rt_mat[:,:]

    plt.imshow(M, cmap='plasma')
    plt.colorbar(orientation='vertical')
    plt.show()