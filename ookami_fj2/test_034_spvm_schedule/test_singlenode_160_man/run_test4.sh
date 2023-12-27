#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 1:30:00
#SBATCH --exclusive

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

size=144
if [ -z "$1" ]; then size=144; else size=$1; fi
echo "N=$size"

for cores in 11 #{0..11}
do
echo "$((cores+1)) cores"
#likwid-pin -c N:0-$cores ./xhpcg --rt=1800 --nx=$size
if [ $2 ]; then
OMP_SCHEDULE="static,"$2 likwid-pin -c N:0-$cores ./xhpcg --rt=300 --nx=$size
else
OMP_SCHEDULE="static" likwid-pin -c N:0-$cores ./xhpcg --rt=300 --nx=$size
fi
done
