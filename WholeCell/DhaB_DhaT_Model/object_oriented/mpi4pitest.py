from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

dicct = {"count": rank}
dict1 = comm.gather(dicct, root=0)
if rank == 0:
	print(dict1)
