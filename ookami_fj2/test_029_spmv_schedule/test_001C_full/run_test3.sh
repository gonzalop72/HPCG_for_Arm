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

for sch in 646 #4 88 176 264 352 440 528 616 645 704 792 880 968 1760 1936 3872 7744 
do
echo "$((sch)) schedule"
OMP_SCHEDULE="static,"$sch likwid-pin -c N:0-11 ./xhpcg --rt=60 --nx=$size
done
