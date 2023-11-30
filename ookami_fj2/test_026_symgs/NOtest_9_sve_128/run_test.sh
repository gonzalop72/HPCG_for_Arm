#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 0:45:00

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

size=144
if [ -z "$1" ]; then size=144; else size=$1; fi
echo "N=$size"

for cores in {0..11}
do
echo "$cores cores"
likwid-pin -c N:0-$cores ./xhpcg --rt=60 --nx=$size
done
