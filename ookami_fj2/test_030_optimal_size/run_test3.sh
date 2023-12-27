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


for size in 16 32 64 96 104 112 120 128 136 144 152 160 168 176 184 192
do
echo "$((size)) size"
likwid-pin -c N:0-11 ./xhpcg --rt=60 --nx=$size
done
