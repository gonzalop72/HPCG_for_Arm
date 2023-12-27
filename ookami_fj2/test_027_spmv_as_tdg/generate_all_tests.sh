#!/bin/bash -l
#-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
# export OMP_SCHEDULE="static" 

#./create_test.sh test_1_ref "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_2_code_ref "-DENABLE_MG_COUNTERS -DHPCG_MAN_OPT_SPMV_UNROLL -DTEST_SPMV_AS_TDG_REF"
#./create_test.sh test_3            "-DENABLE_MG_COUNTERS -DHPCG_MAN_OPT_SPMV_UNROLL  -DTEST_SPMV_AS_TDG"
#export OMP_WAIT_POLICY=active
#./create_test.sh test_4_activewait "-DENABLE_MG_COUNTERS -DHPCG_MAN_OPT_SPMV_UNROLL  -DTEST_SPMV_AS_TDG"
export FLIB_HPCFUNC=TRUE
export FLIB_BARRIER=HARD
./create_test.sh test_6_hardware_wait "-Kparallel -Kocl -Nfjomplib -DENABLE_MG_COUNTERS -DHPCG_MAN_OPT_SPMV_UNROLL  -DTEST_SPMV_AS_TDG"
