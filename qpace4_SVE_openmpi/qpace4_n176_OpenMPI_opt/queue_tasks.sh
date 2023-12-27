#!/bin/bash -l

#scl enable gcc-toolset-10 bash

#module load mpi/openmpi-aarch64

sbatch --partition dev --time=00:30:00  ./sbatch_script.sh
