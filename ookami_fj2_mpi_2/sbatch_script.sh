#!/bin/bash -l

for threads in 10 11 12
do
for size in 160 176
do

for nodes in {32..1..-1} # {1..8}
do
partition_param="--partition short "
if [[ $nodes -gt 1 ]]
then
	partition_param=" --partition short "
fi
for numa in 4  # {1..4}	#numa domain = task-per-node
do
	tasks=$(($nodes*$numa))

	echo "tasks = $(hostname), $partition_param : $tasks"
	sbatch --wait $partition_param -n $tasks -N $nodes --exclusive --ntasks-per-node=$numa --time=0-00:30:00  run_mpi.sh $size $threads 
	#--mem=10gb
	#--constraint=hwperf --cpu-freq=2400000
done
done

done

