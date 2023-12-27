#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 0:45:00

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

for cores in {0..11}
do
echo "$cores cores"
likwid-pin -c N:0-$cores ./xhpcg --rt=60 --nx=144
done
