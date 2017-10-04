#  Before you use this file, install mpi4py on your computer
#  anaconda users do this like that:    conda install mpi4py
#
#  in a command window execute then the following command
#
#  mpiexec -n 2 python ~/Desktop/inspect.py
#  (replace "Desktop" by the path o this file and replace the number of
#  processors from 2 to the number you want.)


from mpi4py import MPI
from scipy import *

comm=MPI.COMM_WORLD

rank=comm.Get_rank()

np=comm.size

if rank == 0:
    msg = "hello World"
    print("I am CPU 0 and sends msg hello world to CPU 1", " size ", np)
    comm.send(msg, dest=1, tag=1)  
    print("now sent it")
    
elif rank == 1: 
    print("I am CPU 1 and got this from CPU 0")
    msg = comm.recv(source = 0, tag=1)
    print(msg)