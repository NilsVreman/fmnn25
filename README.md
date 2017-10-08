# Project 1: Splines
# Project 2: Optimiziation
# Project 3: MPI

* To be able to run Main2Cores and Main3Cores install mpi4py

* If you have anaconda you can run the following in the terminal(Doesn't work on Windows):

```
 conda install --channel mpi4py mpich mpi4py
 ```

*  To run the files enter the folder in which the files are (project3/src) and write the following in the terminal:

```
mpiexec -n 2 python Main2Cores.py

```

* Or

```
mpiexec -n 3 python Main3Cores.py

```
