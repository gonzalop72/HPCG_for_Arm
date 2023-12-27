#!/bin/bash -l

#scl enable gcc-toolset-10 bash

#module load mpi/mpich-aarch64

sbatch --partition qp4 --time=00:30:00  ./sbatch_script.sh
