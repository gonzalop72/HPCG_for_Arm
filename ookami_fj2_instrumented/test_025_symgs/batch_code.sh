#!/bin/bash -l
export OMP_SCHEDULE="static"
./run_test2.sh $1

