#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 0:45:00
#SBATCH --exclusive

#copied from /lustre/home/gapinzon/arm_code/HPCG_for_Arm/ookami_fj2/test_022_ddot/test1_sve

module purge
module load slurm

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

for iter in {0..2}
do
for cores in 11 #{0..11}
do
echo "$cores cores"
OMP_SCHEDULE="static,720" likwid-pin -c N:0-$cores ./xhpcg --rt=300 --nx=176
done
done
