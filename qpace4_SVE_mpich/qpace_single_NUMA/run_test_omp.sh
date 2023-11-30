#!/bin/bash -l

for n in {1..12}
do
export OMP_NUM_THREADS=$n
export OMP_PLACES=cores
export OMP_PROC_BIND="close"
 ../bin/xhpcg
done
