#!/bin/bash -l

module load gnuplot

count=1
lst=""
for path in ../test*
do
	#echo $path
	file=$(basename $path)
	file=${file//"_"/"-"}
	echo $file.csv
	#print_results_awk.sh $path/*.txt | sort -t';' -k3 -n > data_$count.csv
	print_results_awk.sh $path/*.txt | sort -t';' -k3 -n > $file.csv
	count=$(($count+1))
	lst="$lst $file.csv"
done

#set terminal png size 1024,768 enhanced font ,12
echo "set terminal svg size 1024,768 background \"#00FFFFFF\"
set output 'performance_mg.svg'
set xlabel 'cores/threads'
set xrange [1:]
set yrange [0:]
#set logscale xß
set key autotitle columnhead
set datafile separator \";\"

list = \"$lst\"
set output 'performance_SPMV.svg'
set ylabel 'Performance SPPMV [GFlops/s]'
plot for [file in list] file u 3:8 w linespoints title file

set output 'performance_mg.svg'
set ylabel 'Performance MG [GFlops/s]'
plot for [file in list] file u 3:9 w linespoints title file

set output 'performance_SYMGS.svg'
set ylabel 'Performance SYMGS [GFlops/s]'
plot for [file in list] file u 3:20 w linespoints title file" > plot_graph_gen.plot

#gnuplot plot_graph.plot
gnuplot plot_graph_gen.plot
