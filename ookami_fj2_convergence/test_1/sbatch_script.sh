#list=(1 2 4 5 8 10)  #400
#list=(1 2 3 4 6 8 9 18 27) #432
#list=(1 2 4 5 8 10) #480
#list=(1 2 3 4 6 9) #360
list=( 1 2 4 5 8 10 20) #320
listxy=( 1 ) # (1 2 4 5)
# 1 DIM p:artition
ndimx=320
ndimy=320
ndimz=320

#NOTES: On ARM the subdomain must be a multiple of 8 to work 

for zdiv in "${list[@]}"
do
nz=$(($ndimz / zdiv))
for xdiv in  "${listxy[@]}"
do
	nx=$(($ndimx / xdiv))
	for ydiv in "${listxy[@]}"
	do
		ny=$(($ndimy / ydiv))
		#nodes=$((($subdomains*$subdomains*$subdomains+3)/4))
		tasks=$(($xdiv*$ydiv*$zdiv))

		nodes=$((($tasks+19)/20))
		partition_param="--partition short"
		if [[ $nodes -gt 1 ]]
		then
			partition_param=" --partition short "
		fi

		echo "domain=($ndimx,$ndimy,$ndimz), subdomains=($xdiv,$ydiv,$zdiv), tasks=$tasks, nodes=$nodes"
		echo "tasks = $(hostname), $partition_param : $tasks,$nodes"
		sbatch $partition_param --ntasks=$tasks --nodes=$nodes --mem=0 --exclusive  --time=0-00:30:00  job_mpi.sh $nx $ny $nz $xdiv $ydiv $zdiv 
	#--mem=10gb
	done
done
done
