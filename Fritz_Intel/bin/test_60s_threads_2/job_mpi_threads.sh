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

#codes per node / processes per node
export OMP_NUM_THREADS=18
export OMP_PLACES=cores
export OMP_PROC_BIND="close"

for threads in 1 2 4 6 8 10 12 14 16 18
do
export OMP_NUM_THREADS=$threads
mpirun -genvall -print-rank-map ../xhpcg -t60
done

