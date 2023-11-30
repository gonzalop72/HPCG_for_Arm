#!/bin/bash -l
export OMP_SCHEDULE="static,144"
./run_test2.sh $1

