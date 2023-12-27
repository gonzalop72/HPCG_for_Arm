#!/bin/bash -l

module load likwid/5.1.1
module load arm-modules/22.1 

for size in {160..192..16}
do
likwid-pin -c N:0-11 ../bin/xhpcg --rt=300 --nx=$size
done
