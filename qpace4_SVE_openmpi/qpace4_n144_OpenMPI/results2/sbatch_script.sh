#!/bin/bash -l

for nodes in 9 10 11 12 13 14 15 16  #  {1..8}
do
partition_param="--partition qp4"
if [[ $nodes -gt 1 ]]
then
	partition_param=" --partition qp4 "
fi
for numa in 4 # {1..4}	#numa domain = task-per-node
do
	tasks=$(($nodes*$numa))

	echo "tasks = $(hostname), $partition_param : $tasks"
	sbatch $partition_param -n $tasks -N $nodes --ntasks-per-node=$numa --time=0-00:30:00  run_mpi.sh 
	#--mem=10gb
	#--constraint=hwperf --cpu-freq=2400000
done
done

