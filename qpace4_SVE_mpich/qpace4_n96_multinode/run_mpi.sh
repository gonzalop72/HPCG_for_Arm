#!/bin/bash -l

#export I_MPI_PIN=1
#export I_MPI_PIN_DOMAIN="12:compact"
#export I_MPI_PIN_PROCESSOR_LIST="allcores"

export OMP_NUM_THREADS=12
export OMP_PLACES=cores
export OMP_PROC_BIND="close"

module load mpi/mpich-aarch64

#mpirun - openmp
#mpirun --map-by socket:pe=10 --bind-to core --report-bindings -N 1 -n 1 ./xhpcg


#mpich - hydra
#testing mapping - mpich 
#HYDRA_TOPO_DEBUG=1 mpiexec -n 8 -bind-to core:10 -map-by numa /bin/true | sort -k 2 -n
mpiexec -genvall -bind-to core:12 -map-by numa -prepend-rank ../bin/xhpcg -t300 --rt=300 
