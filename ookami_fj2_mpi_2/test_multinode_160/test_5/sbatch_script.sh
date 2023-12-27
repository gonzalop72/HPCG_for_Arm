#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p long
#S---BATCH --ntasks-per-node=48
#SBATCH --ntasks-per-node=1
#SBATCH -t 23:30:00
#S---BATCH --exclusive

export OMP_WAIT_POLICY=active 
size=160

for threads in 12 #10 11 12
do
for iter in {1..3}
do
#single process: reference value
partition_param="--partition short "
#sbatch $partition_param -n 1 --exclusive --time=0-01:00:00  run_mpi.sh $size $threads 
done

for nodes in {7..1..-1} #{15..1..-1} #32 30 28 27 26 25 24 22 21 20 18 17 16 #{32..16..-1} #25 26 27 28 30 32 #{1..24} #{32..1..-1} # {1..8}
do
partition_param="--partition short "
if [[ $nodes -gt 1 ]]
then
	partition_param=" --partition short "
fi
for numa in 4  # {1..4}	#numa domain = task-per-node
do
	tasks=$(($nodes*$numa))

    dowait="--wait"
    #if [ "$nodes" -ge 12 ]; then dowait="--wait"; fi
    #if [ "$nodes" -eq 9 ]; then dowait="--wait"; fi

	for iter in {1..2}
	do
	echo "tasks = $(hostname), $partition_param : $tasks"
	sbatch $partition_param -n $tasks -N $nodes --exclusive --ntasks-per-node=$numa --time=0-01:00:00  run_mpi.sh $size $threads 
	done
	sbatch $dowait $partition_param -n $tasks -N $nodes --exclusive --ntasks-per-node=$numa --time=0-01:00:00  run_mpi.sh $size $threads 

	#--mem=10gb
	#--constraint=hwperf --cpu-freq=2400000
done
done
done
