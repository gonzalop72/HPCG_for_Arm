#!/bin/bash -l
mkdir $1
mkdir $1/compilation_results
cd ..
make clean
EXT_HPCG_OPTS=$2 make -j 10
cd -
mv ../*.lst $1/compilation_results
mv ../bin/xhpcg $1
cp ../setup/Make.OOKAMI_OMP_FJ $1
cp run_test.sh $1
cd $1
sbatch run_test.sh $3
cd -
