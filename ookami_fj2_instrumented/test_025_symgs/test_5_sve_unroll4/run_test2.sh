#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 0:45:00

module load fujitsu/compiler/4.7  
module load likwid/5.2.2
module load arm-modules/22.1

nodes=144
lvl=0
for cores in {6..12}
do
group=$1
#for group in MEM MEM_DP L2 FLOPS_DP
#do
    cores1=$((cores-1))
    echo "cores: $cores"
    likwid-perfctr -C N:0-$cores1 -g $group -m -o markers_${cores}_${lvl}_${group}_%j_%h_%p.txt -O ./xhpcg --rt=60 --nx=$nodes --lvt=${lvl}
#done
done
