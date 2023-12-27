#!/bin/bash -l
export OMP_SCHEDULE="static,20750"
./run_test2.sh $1

