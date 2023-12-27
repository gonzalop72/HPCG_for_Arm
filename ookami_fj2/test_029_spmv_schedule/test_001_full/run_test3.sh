#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 1:30:00

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

size=144
if [ -z "$1" ]; then size=144; else size=$1; fi
echo "N=$size"

OMP_SCHEDULE="static" likwid-pin -c N:0-11 ./xhpcg --rt=60 --nx=$size

for sch in 432 #756 828 #792 864 #432 504 648 720 #4 36 72 144 288 576 1296 5184 
do
echo "$((sch)) schedule"
OMP_SCHEDULE="static,"$sch likwid-pin -c N:0-11 ./xhpcg --rt=60 --nx=$size
done
