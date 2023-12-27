#/bin/bash -l

sbatch -N 1 -n 48 -t 1:00:00 -p short ./run_test.sh 
