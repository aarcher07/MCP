#!/bin/bash
#SBATCH --account=b1114     
#SBATCH --partition=b1114  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:04:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=sample_job
#SBATCH --array=0-39    -N1 tmp
#SBATCH --output=outlog
#SBATCH --error=errlog

module load gcc/6.4.0
module load python/anaconda
module load mpi4py

mpirun -n 8 python active_subspaces.py 1:3. 1e1