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
export OMP_NUM_THREADS=$1 
mpirun --map-by socket:pe=$1 --bind-to core --report-bindings xhpcg --nx=96 --ny=96 --nz=192 --npx=2 --npy=2 --npz=1 --rt=1800