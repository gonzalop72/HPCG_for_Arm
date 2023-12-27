#!/bin/bash -l

unset OMP_SCHEDULE
for size in {96..192..16}
do
        #sbatch -p singlenode likwid-pin -c N:0-17 bin/xhpcg --rt=300 --nx=$size
        sbatch -p singlenode ./single_test.sh $size
done

#for sched in "static,1" "static,10" "static,100" "static,1000" "dynamic,1" "dynamic,10"  "dynamic,100" "dynamic,1000"
#do
#for size in {96..192..16}
#do
#	export OMP_SCHEDULE=$sched
#	#sbatch -p singlenode likwid-pin -c N:0-17 bin/xhpcg --rt=300 --nx=$size
#	sbatch -p singlenode ./single_test.sh $size
#done
#done
