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

#NOTES: --rt= instead of -t
#mpirun -genvall -print-rank-map ../bin/xhpcg --rt=300
for lvl in 2 # {1..4}
do
mpirun -genvall -print-rank-map likwid-perfctr -g MEM_DP -m -o markers_${lvl}_FLOPSXX_%j_%h_%p.txt ../bin/xhpcg --rt=300 -n192 --lvt=${lvl}
#mpirun -genvall -print-rank-map likwid-perfctr -g FLOPS_DP -m -o markers_${lvl}_FLOPS_%j_%h_%p.txt ../bin/xhpcg --rt=300 -n192 --lvt=${lvl}
#mpirun -genvall -print-rank-map likwid-perfctr -g MEM -m -o markers_${lvl}_MEM_%j_%h_%p.txt ../bin/xhpcg --rt=300 -n192 --lvt=${lvl}
#mpirun -genvall -print-rank-map likwid-perfctr -g L3 -m -o markers_${lvl}_L3_%j_%h_%p.txt ../bin/xhpcg --rt=300 -n192 --lvt=${lvl}
done