#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J train           # Job name
#SBATCH -o train.o%j       # Name of stdout output file
#SBATCH -e train.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 4               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 9:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=malb23@lehigh.edu

source /scratch1/99999/malb23/model/python-env1/bin/activate

scontrol show hostnames $SLURM_NODELIST > /tmp/hostfile

cat /tmp/hostfile

python -m torch.distributed.launch --nproc_per_node=4 train.py 

# -----------------------------------------------------------------------------
