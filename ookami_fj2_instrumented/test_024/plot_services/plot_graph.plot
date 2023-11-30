#set terminal png size 1024,768 enhanced font ,12
set terminal svg size 1024,768 background "#00FFFFFF"
set output 'performance.svg'
set xlabel 'cores/threads'
set xrange [1:]
set yrange [0:]
set ylabel 'Performance [GFlops/s]'
#set logscale xß
set key autotitle columnhead
set datafile separator ";"

plot for [i = 1:3] "data\_".i.".csv" u 3:8 w linespoints title 'performance SPMV\_'.i
