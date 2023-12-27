#list=(1 2 4 5 8 10)  #400
list=(1 2 4 5 8 10) #480
#360 : 1 2 3 4 5 6
#320 : 1 2 4
# 1 DIM partition
ndimx=480
ndimy=480
ndimz=480

for zdiv in "${list[@]}"
do
nz=$(($ndimz / zdiv))
for xdiv in  "${list[@]}"
do
	nx=$(($ndimx / xdiv))
	for ydiv in "${list[@]}"
	do
		ny=$(($ndimy / ydiv))
		#nodes=$((($subdomains*$subdomains*$subdomains+3)/4))
		tasks=$(($xdiv*$ydiv*$zdiv))

		nodes=$((($tasks+23)/24))
		partition_param=""
		if [[ $nodes -gt 1 ]]
		then
			partition_param=" --partition multinode "
		fi

		echo "domain=($ndimx,$ndimy,$ndimz), subdomains=($xdiv,$ydiv,$zdiv), tasks=$tasks, nodes=$nodes"
		echo "tasks = $(hostname), $partition_param : $tasks,$nodes"
		sbatch $partition_param --ntasks=$tasks --nodes=$nodes --mem=0 --exclusive --constraint=hwperf --cpu-freq=2400000  --time=0-00:30:00  job_mpi.sh $nx $ny $nz $xdiv $ydiv $zdiv
	#--mem=10gb
	done
done
done
