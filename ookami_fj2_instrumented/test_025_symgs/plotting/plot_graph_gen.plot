set terminal svg size 1024,768 background "#00FFFFFF"
set output 'performance_mg.svg'
set xlabel 'cores/threads'
set xrange [1:]
set yrange [0:]
#set logscale xß
set key autotitle columnhead
set datafile separator ";"

list = " test-1-noopt.csv test-2-neon.csv test-3e-sve.csv test-3-sve.csv test-4-sve.csv test-5-sve-unroll4.csv"
set output 'performance_SPMV.svg'
set ylabel 'Performance SPPMV [GFlops/s]'
plot for [file in list] file u 3:8 w linespoints title file

set output 'performance_mg.svg'
set ylabel 'Performance MG [GFlops/s]'
plot for [file in list] file u 3:9 w linespoints title file

set output 'performance_SYMGS.svg'
set ylabel 'Performance SYMGS [GFlops/s]'
plot for [file in list] file u 3:20 w linespoints title file
