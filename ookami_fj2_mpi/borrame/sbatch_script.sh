#!/bin/bash -l

for nodes in {8..8} # {1..32}
do
partition_param="--partition short "
if [[ $nodes -gt 1 ]]
then
	partition_param=" --partition short "
fi
for numa in 1  # {1..4}	#numa domain = task-per-node
do
	tasks=$(($nodes*$numa))

	echo "tasks = $(hostname), $partition_param : $tasks"
	#sbatch $partition_param -n $tasks -N $nodes --ntasks-per-node=$numa --time=0-00:30:00  run_mpi.sh 
	sbatch $partition_param -n $tasks -N 2 --ntasks-per-node=4 --time=0-00:30:00  run_mpi.sh 
	#--mem=10gb
	#--constraint=hwperf --cpu-freq=2400000
done
done

