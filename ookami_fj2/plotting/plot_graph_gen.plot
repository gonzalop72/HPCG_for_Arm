set terminal svg size 1024,768 background "#00FFFFFF"
set output 'performance_mg.svg'
set xlabel 'cores/threads'
set xrange [1:]
set yrange [0:]
#set logscale xß
set key autotitle columnhead
set datafile separator ";"

list = " test-NEW-1-static.csv test-NEW-4-unroll4-static-18.csv test-NEW-5-unroll4-static-36.csv test-NEW-6-unroll4-static-72.csv test-NEW-7-unroll4-static-144.csv test-NEW-8-unroll4-static-144.csv test-NEW-9-unroll4-static-144.csv test-NEW-unroll2-static.csv test-NEW-unroll4-static.csv"
set output 'performance_SPMV.svg'
set ylabel 'Performance SPPMV [GFlops/s]'
plot for [file in list] file u 3:8 w linespoints title file

set output 'performance_mg.svg'
set ylabel 'Performance MG [GFlops/s]'
plot for [file in list] file u 3:9 w linespoints title file
