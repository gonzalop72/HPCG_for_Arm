make clean
EXT_HPCG_OPTS="-DUNROLLING_6_B" make -j20
OMP_SCHEDULE="STATIC,560" likwid-pin -c N:0-11 bin/xhpcg --nx=96 --rt=10
