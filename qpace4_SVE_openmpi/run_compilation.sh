#!/bin/bash -l

scl enable gcc-toolset-10 bash
module load mpi/openmpi-aarch64
module load armpl/22.0.2_gcc-10.2

make clean
make

