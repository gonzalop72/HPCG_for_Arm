#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p long
#S---BATCH --ntasks-per-node=48
#SBATCH --ntasks-per-node=1
#SBATCH -t 11:30:00
#S---BATCH --exclusive

export OMP_WAIT_POLICY=active 
size=160

for threads in 12
do

#single process: reference value
partition_param="--partition short "
sbatch $partition_param -n 1 --exclusive --time=0-00:30:00  run_mpi.sh $size $threads 

for nodes in {1..32} #25 26 27 28 30 32 #{1..24} #{32..1..-1} # {1..8}
do
partition_param="--partition short "
if [[ $nodes -gt 1 ]]
then
	partition_param=" --partition short "
fi
for numa in 4  # {1..4}	#numa domain = task-per-node
do
	tasks=$(($nodes*$numa))

    dowait=""
    #if [ "$nodes" -ge 12 ]; then dowait="--wait"; fi

	echo "tasks = $(hostname), $partition_param : $tasks"
	sbatch $dowait $partition_param -n $tasks -N $nodes --exclusive --ntasks-per-node=$numa --time=0-00:30:00  run_mpi.sh $size $threads 
	#--mem=10gb
	#--constraint=hwperf --cpu-freq=2400000
done
done
done
