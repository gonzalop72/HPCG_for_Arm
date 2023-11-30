#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 0:45:00

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

#NOTES: $1 nx size
#export OMP_SCHEDULE="static,144"
size=144
if [ -z "$1" ]; then size=144; else size=$1; fi
echo "N=$size"

for cores in {1..12}
do
cores1=$((cores-1))
echo "$cores1 cores"
likwid-pin -c N:0-$cores1 ./xhpcg --rt=60 --nx=$size
done
