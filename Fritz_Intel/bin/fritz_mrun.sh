#!/bin/sh -l

module unload openmpi
module load mkl/2021.4.0  
module load intel/2022.1.0
module load likwid/5.2.1
module load intelmpi/2021.6.0

#mpirun variables
#declare I_MPI_HYDRA_BOOTSTRAP="slurm"
declare I_MPI_PIN=1
declare I_MPI_PIN_DOMAIN="18:compact"
declare I_MPI_PIN_PROCESSOR_LIST="allcores"

#codes per node / processes per node
export OMP_NUM_THREADS=18
export OMP_PLACES=cores
export OMP_PROC_BIND="close"

#optional when using mpiexec.hydra
#export KMP_AFFINITY=granularity=fine,compact,1,0

# Number of MPI tasks
#SBATCH -n 1
#
# Number of cores per task
#SBATCH -c 32
#
# Runtime of this jobs is less then 12 hours.
#SBATCH --time=12:00:00
#  -n == -nodes = -np : compute nodes
#  -ppn == processes per node
#mpirun -n 1 -ppn 1 -verbose -host optane1 ../bin/xhpcg_skx -t60
	mpirun -genvall -n -print-rank-map  --constraint=hwperf --time=0:30:00 ../../bin/xhpcg_skx -t60 -n192
#srun -N 1 -n 1 --ntasks-per-node 1 --constraint=hwperf --time=16:00:00 ../../bin/xhpcg_skx -t60
