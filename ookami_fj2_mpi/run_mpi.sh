#!/bin/bash -l

#export I_MPI_PIN=1
#export I_MPI_PIN_DOMAIN="12:compact"
#export I_MPI_PIN_PROCESSOR_LIST="allcores"

export OMP_NUM_THREADS=10
export OMP_PLACES=cores
export OMP_PROC_BIND="close"
export OMP_WAIT_POLICY=active

module load slurm
module load fujitsu/compiler/4.7

#mpirun - openmp
mpirun --map-by socket:pe=12 --bind-to core --report-bindings bin/xhpcg --nx=144 --rt=60
