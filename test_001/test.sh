#!/usr/bin/sh -l

for n in {0..17}
do
	likwid-pin -c N:0-$n bin/xhpcg --nx=192 --rt=60 #--lvt=1
done
