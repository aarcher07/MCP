#!/bin/bash
#SBATCH --account=b1020
#SBATCH --partition=buyin-dev
#SBATCH --nodes=1
#SBATCH --array=1
#SBATCH --ntasks=10
#SBATCH --time=00:04:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrearcher2017@northwestern.edu
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=sample_job
#SBATCH --output=outlog
#SBATCH --error=errlog

module purge all
module load texlive/2020

mpirun -n 10 python active_subspaces.py 1:3 1e1 rsampling 0