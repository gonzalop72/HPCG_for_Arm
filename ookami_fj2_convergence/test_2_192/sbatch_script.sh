#!/bin/bash -l

#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 9
#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 10
#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 11
#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 12

#reference : 1 task
sbatch --partition short --ntasks=1 --nodes=1 --mem=0 --exclusive  --time=0-01:30:00  run_mpi_single.sh 12

#4 tasks
sbatch --partition short --ntasks=4 --nodes=1 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 10
sbatch --partition short --ntasks=4 --nodes=1 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 11
sbatch --partition short --ntasks=4 --nodes=1 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 12
