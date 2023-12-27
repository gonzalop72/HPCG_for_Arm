#!/bin/bash -l

module load gnuplot

#for path in 
print_results_awk.sh ../test_10_baseline/*.txt | sort -t';' -k3 -n > data_1.csv
print_results_awk.sh ../test_9_performance/*.txt | sort -t';' -k3 -n > data_2.csv
print_results_awk.sh ../test_10_performance/*.txt | sort -t';' -k3 -n > data_3.csv

gnuplot plot_graph.plot