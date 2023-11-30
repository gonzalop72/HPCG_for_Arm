#!/bin/bash -l

#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 9
#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 10
#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 11
#sbatch --partition short --ntasks=4 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 12

sbatch --partition short --ntasks=8 --nodes=2 --mem=0 --exclusive  --time=0-01:30:00  run_mpi.sh 11
