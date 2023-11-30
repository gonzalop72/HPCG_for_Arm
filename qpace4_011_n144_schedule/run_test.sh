#!/bin/bash -l


for sched in "static,1" "static,10" "static,100" "static,1000" "dynamic,1" "dynamic,10"  "dynamic,100" "dynamic,1000"
do
for size in 144 # {96..192..16}
do
	export OMP_SCHEDULE=$sched
	likwid-pin -c N:0-11 bin/xhpcg --rt=300 --nx=$size
	#sbatch -p singlenode ./single_test.sh $size
done
done

