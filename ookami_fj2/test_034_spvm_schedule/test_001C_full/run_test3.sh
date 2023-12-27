#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH --ntasks-per-node=48
#SBATCH -t 1:30:00

module load fujitsu/compiler/4.7  
module load likwid/5.2.2

size=144
if [ -z "$1" ]; then size=144; else size=$1; fi
echo "N=$size"

OMP_SCHEDULE="static" likwid-pin -c N:0-11 ./xhpcg --rt=60 --nx=$size

if [ $size -eq 144 ]; then
	list=(0 4 36 72 144 288 432 504 576 648 720 756 792 828 864 1296 5184)
elif [ $size -eq 160 ]; then
	list=(0 4 80 160 240 320 400 480 560 640 680 720 760 800 1600 3200 6400)
elif [ $size -eq 176 ]; then
	list=(0 4 88 176 264 352 440 528 616 645 704 720 792 880 968 1760 1936 3872 7744)
else #192
	list=(0 4 96 192 288 384 480 576 672 720 768 864 912 960 1056 1920 2304 4608 9216)
fi

for sch in "${list[@]}";
do
echo "$((sch)) schedule"
OMP_SCHEDULE="static,"$sch likwid-pin -c N:0-11 ./xhpcg --rt=60 --nx=$size
done
