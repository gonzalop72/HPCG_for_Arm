#!/bin/bash -l
for size in {96..192..16}
do
likwid-pin -c N:0-11 ../bin/xhpcg --rt=300 --nx=$size
done
