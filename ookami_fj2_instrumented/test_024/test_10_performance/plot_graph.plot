#set terminal png size 1024,768 enhanced font ,12
set terminal svg
set output 'performance.svg'
set xlabel 'cores/threads'
set xrange [1:]
set yrange [0:]
set ylabel 'Performance [GFlops/s]'
#set logscale xß
set key autotitle columnhead
set datafile separator ";"

plot 'data.csv' u 3:6 w linespoints title 'performance DDOT', \
    'data.csv' u 3:8 w linespoints title 'performance SPMV'
