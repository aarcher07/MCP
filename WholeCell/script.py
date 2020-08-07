from mpi4py import MPI
import numpy as np
import sympy as sp
import string
alphabet = list(string.ascii_lowercase)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def print_data(data = None):

    k = 2
    if rank == 0:
        listt = [[i]*2 for i in range(size)]
        data = list(range(size))
    else:
        data = None
        listt = None
    data = comm.scatter(data, root=0)
    listt = comm.scatter(listt, root = 0)
    answer =comm.reduce((data,listt),op=MPI.MAXLOC)

    print(str(rank) + ' ' + str(answer))
    #if rank == 0:
    #    print(answer)


if __name__ == '__main__':
    print_data()


