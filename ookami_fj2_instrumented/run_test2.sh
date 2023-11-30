#/bin/bash -l

nodes=144
lvl=0
for cores in {1..12}
do
for group in MEM MEM_DP L2 FLOPS_DP
do
    cores1=$((cores-1))
    echo "cores: $cores"
    likwid-perfctr -C N:0-$cores1 -g $group -m -o markers_${cores}_${lvl}_${group}_%j_%h_%p.txt -O bin/xhpcg --rt=60 --nx=$nodes --lvt=${lvl}
done
done
