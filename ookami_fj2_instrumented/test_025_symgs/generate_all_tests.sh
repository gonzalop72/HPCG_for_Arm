#!/bin/bash -l
#-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
export OMP_SCHEDULE="static" 
#./create_test.sh test_1_noopt "-DENABLE_MG_COUNTERS"
#./create_test.sh test_2_neon "-DENABLE_MG_COUNTERS -DHPCG_USE_NEON" NOTES: NEON DOES NOT EXISTS ON FCC
#./create_test.sh test_3_sve "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"

#./create_test.sh test_NEW_1_static "-Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
#./create_test.sh test_NEW_unroll2_static "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_2_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
#./create_test.sh test_NEW_unroll4_static "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
# export OMP_SCHEDULE="static,18" 
# ./create_test.sh test_4_unroll4_static_18 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
# export OMP_SCHEDULE="static,36" 
# ./create_test.sh test_5_unroll4_static_36 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
# export OMP_SCHEDULE="static,72" 
# ./create_test.sh test_6_unroll4_static_72 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
# export OMP_SCHEDULE="static,144" 
# ./create_test.sh test_7_unroll4_static_144 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON"
# export OMP_SCHEDULE="static"
# ./create_test.sh test_8_unroll4_static_144 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON -DHPCG_USE_NEON"
# ./create_test.sh test_9_unroll4_static_144 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON -DHPCG_USE_SVE"
# export OMP_SCHEDULE="static" 
# ./create_test.sh test_NEW_B1_unroll4_static "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON" 160
# export OMP_SCHEDULE="static,160" 
# ./create_test.sh test_NEW_B2_unroll4_static_160 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON" 160

#./create_test.sh test_3e_sve "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_4_sve "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_5_sve_unroll4 "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_6_sve_unroll4_3 "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE" #4 unrolling with omp parallel

export OMP_WAIT_POLICY=active
./create_test.sh test_8_activewait "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"