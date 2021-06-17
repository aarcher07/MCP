from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

dicct = {"count": rank}
dict1 = comm.gather(dicct, root=0)
n = 2
split = np.cumsum(size*[n])
if rank == 0:
	array = np.array(range(n*size))
	print(array)
	array_split = np.split(array,split[:-1])
	print(array_split)
else:
	array_split = None
arrays=comm.scatter(array_split,root=0)
print(arrays)