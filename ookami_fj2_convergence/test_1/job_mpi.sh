#!/bin/bash -l
#script wrapper for mpirun process

export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND="close"
export OMP_WAIT_POLICY=active

module load slurm
module load fujitsu/compiler/4.7
echo "params: --nx=$1 --ny=$2 --nz=$3 --npx=$4 --npy=$5 --npz=$6"

mpirun -map-by node:PE=2 --bind-to core --report-bindings ../bin/xhpcg --rt=60 --nx=$1 --ny=$2 --nz=$3 --npx=$4 --npy=$5 --npz=$6
