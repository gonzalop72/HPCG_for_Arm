#!/bin/bash -l
#-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
# export OMP_SCHEDULE="static" 

#./create_test.sh test_010_ref ""
#./create_test.sh test_011_ref_sve "-DHPCG_USE_SVE"
#OMP_WAIT_POLICY=active ./create_test.sh test_012_full "-DHPCG_USE_SVE -DHPCG_MAN_OPT_DDOT -DDDOT_2_UNROLL -DWAXPBY_AUTO_OPT -DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL"

#./create_test.sh test_020_ref ""
#./create_test.sh test_021_ref_sve "-DHPCG_USE_SVE"
#OMP_WAIT_POLICY=active ./create_test.sh test_022_full "-DHPCG_USE_SVE -DHPCG_MAN_OPT_DDOT -DDDOT_2_UNROLL -DWAXPBY_AUTO_OPT -DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL"
./create_test.sh test_022B_full ""
./create_test.sh test_023_full_struc "-DDMAN_OPTIMAL_SPMV_STRUCTURE"