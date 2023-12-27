for nodes in {2..2}
do
partition_param="--partition singlenode "
if [[ $nodes -gt 1 ]]
then
	partition_param=" --partition multinode "
fi
for numa in 1 # {1..4}	#numa domain = task-per-node
do
	tasks=$(($nodes*$numa))

	echo "tasks = $(hostname), $partition_param : $tasks"
	sbatch $partition_param -n $tasks -N $nodes --ntasks-per-node=$numa --constraint=hwperf --cpu-freq=2400000 --time=0-00:30:00  job_mpi.sh 
	#--mem=10gb
done
done

