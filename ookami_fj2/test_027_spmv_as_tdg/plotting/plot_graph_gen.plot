set terminal svg size 1024,768 background "#00FFFFFF"
set output 'performance_mg.svg'
set xlabel 'cores/threads'
set xrange [1:]
set yrange [0:]
#set logscale xß
set key autotitle columnhead
set datafile separator ";"

list = " test-1-ref.csv test-2-code-ref.csv test-3.csv test-4-activewait.csv test-5-activewait.csv"
set output 'performance_SPMV.svg'
set ylabel 'Performance SPPMV [GFlops/s]'
plot for [file in list] file u 3:8 w linespoints title file

set output 'performance_mg.svg'
set ylabel 'Performance MG [GFlops/s]'
plot for [file in list] file u 3:9 w linespoints title file

set output 'performance_gs.svg'
set ylabel 'Performance SymGS [GFlops/s]'
plot for [file in list] file u 3:20 w linespoints title file


