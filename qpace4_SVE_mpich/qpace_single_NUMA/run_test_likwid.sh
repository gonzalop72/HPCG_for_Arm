for n in {1..12}
do
n1=$(($n-1))
likwid-pin -c N:0-$n1 ../bin/xhpcg
done
