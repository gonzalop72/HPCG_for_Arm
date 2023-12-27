#!/bin/bash -l
#script wrapper for mpirun process

#module unload openmpi
#module load mkl/2021.4.0
#module load intel/2022.1.0
#module load likwid/5.2.1
#module load intelmpi/2021.6.0

export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN="18:compact"
export I_MPI_PIN_PROCESSOR_LIST="allcores"
#export I_MPI_PIN_PROCESSOR_LIST="0-17,18-35,36-53"

#export KMP_AFFINITY="sockets,compact,verbose"

#codes per node / processes per node
export OMP_NUM_THREADS=18
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "params: --nx=$1 --ny=$2 --nz=$3 --npx=$4 --npy=$5 --npz=$6"
mpirun -genvall -print-rank-map ../bin/xhpcg -t60 --nx=$1 --ny=$2 --nz=$3 --npx=$4 --npy=$5 --npz=$6
# 
