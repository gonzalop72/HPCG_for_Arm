nodes=1
numa=1
tasks=1
	sbatch -n $tasks -N $nodes --ntasks-per-node=$numa --constraint=hwperf --cpu-freq=2400000 --time=0-00:30:00 -w f1101 job_mpi_threads.sh 

