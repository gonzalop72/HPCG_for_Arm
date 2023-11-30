#!/bin/bash
#SBATCH -N 1
#SBATCH -p long
#SBATCH --ntasks-per-node=48
#SBATCH -t 22:00:00
#SBATCH --exclusive

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

export OMP_WAIT_POLICY=active 

size=144
if [ -z "$1" ]; then size=144; else size=$1; fi
echo "N=$size"

for cores in {11..0}
do
for iter in {0..2}
do
echo "$((cores+1)) cores, $2 schdule"
#likwid-pin -c N:0-$cores ./xhpcg --rt=1800 --nx=$size
if [ $2 ]; then
OMP_SCHEDULE="static,"$2 likwid-pin -c N:0-$cores ./xhpcg --rt=1800 --nx=$size
else
OMP_SCHEDULE="static" likwid-pin -c N:0-$cores ./xhpcg --rt=1800 --nx=$size
fi
done
done