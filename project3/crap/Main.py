from Room_temp import Room_temp
import numpy as np

if __name__ == '__main__':
    length_B_x = 1
    length_B_y = 2
    length_S_x = 1
    length_S_y = 1
    dx = 1/3
    nbr_iter = 10
    w = 0.75
 
    B_rt = Room_temp(length_B_x, length_B_y, dx)
    SL_rt = Room_temp(length_S_x, length_S_y, dx)
    SR_rt = Room_temp(length_S_x, length_S_y, dx)

    # Nbr grid points in different rooms and directions
    B_nbr_pts_x = B_rt.nbr_pts_x
    B_nbr_pts_y = B_rt.nbr_pts_y
    S_nbr_pts_x = SL_rt.nbr_pts_x
    S_nbr_pts_y = SL_rt.nbr_pts_y

    # Createing boundary condition grid vectors / AKA Walls
    B_wallL, B_wallR, B_wallB, B_wallT = B_rt.create_walls()
    S_wallL, S_wallR, S_wallB, S_wallT = SL_rt.create_walls()

    # temperatures corresponding to each wall
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
    #Small right
    SR_valuesL = np.ones(S_nbr_pts_y) #To be calculated
    SR_valuesR = np.ones(S_nbr_pts_y)*40
    SR_valuesB = np.ones(S_nbr_pts_x)*15
    SR_valuesT = np.ones(S_nbr_pts_x)*15
    
    #Add fixed conditions on unshared walls
    #Big
    B_rt.add_dirichlet_cond(B_wallB, B_valuesB)
    B_rt.add_dirichlet_cond(B_wallT, B_valuesT)
    #Small left
    SL_rt.add_dirichlet_cond(S_wallL, SL_valuesL)
    SL_rt.add_dirichlet_cond(S_wallT, SL_valuesT)
    SL_rt.add_dirichlet_cond(S_wallB, SL_valuesB)
    #Small right
    SR_rt.add_dirichlet_cond(S_wallR, SR_valuesR)
    SR_rt.add_dirichlet_cond(S_wallT, SR_valuesT)
    SR_rt.add_dirichlet_cond(S_wallB, SR_valuesB)
    
    #Add initial guess on shared walls for B_rt
    B_rt.add_dirichlet_cond(B_wallL, B_valuesL)
    B_rt.add_dirichlet_cond(B_wallR, B_valuesR)

    #Create u_k matrix
    u_size = SL_rt.nbr_pts_x*SL_rt.nbr_pts_y + SR_rt.nbr_pts_x*SR_rt.nbr_pts_y + B_rt.nbr_pts_x*B_rt.nbr_pts_y 
    u = np.zeros((u_size, nbr_iter))
    #for printing
    #Neumann-Dirichlet Iteration
    for i in range(0, nbr_iter):

        #STEP 1 - calculate temp in room 2
        u_B = B_rt()
        B_rt_mat = B_rt.find_temp_matrix(u_B)
        #Calculate neumann cond for room 1 and 3
        gamma1 = 1*(B_rt_mat[int(len(B_rt_mat)/2):, 0] - B_rt_mat[int(len(B_rt_mat)/2):, 1])
        gamma2 = 1*(B_rt_mat[:int(len(B_rt_mat) / 2 + 1), -1] - B_rt_mat[:int(len(B_rt_mat) / 2 + 1), -2])

        #Add the neumann condition on gamma1 to wall
        SL_rt.add_neumann_cond(S_wallR, gamma1)
        SR_rt.add_neumann_cond(S_wallL, gamma2)

        #STEP 2 - calculate the temp in room 1 and 3
        u_SL = SL_rt()
        u_SR = SR_rt()

        #STEP 3 - relaxation and then adding dirichlet cond to room 2
        if i == 0:
            u[:, i] = np.append(u_SL, np.append(u_B, u_SR))
        else:
            u[:, i] = np.append(u_SL, np.append(u_B, u_SR))*w + (1-w)*u[:, i-1]

        #Add new dirichlet temp to Big room
        SL_rt_mat = SL_rt.find_temp_matrix(u[0:len(u_SL), i])
        SR_rt_mat = SL_rt.find_temp_matrix(u[-len(u_SR):, i])

        B_rt.add_dirichlet_cond(B_wallL[int(len(B_wallL) / 2):, :], SL_rt_mat[:,-1])
        B_rt.add_dirichlet_cond(B_wallR[:int(len(B_wallR) / 2 + 1), :], SR_rt_mat[:, 0])


    print('Temperature in room 1 is now: \n', SL_rt_mat)
    print('Temperature in room 2 is now: \n', B_rt.find_temp_matrix(B_rt()))
    print('Temperature in room 1 is now: \n', SR_rt_mat)
