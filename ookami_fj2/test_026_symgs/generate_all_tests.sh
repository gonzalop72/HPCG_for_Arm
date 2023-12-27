#!/bin/bash -l
#-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
export OMP_SCHEDULE="static" 
#./create_test.sh test_1_noopt "-DENABLE_MG_COUNTERS"
##./create_test.sh test_2_neon "-DENABLE_MG_COUNTERS -DHPCG_USE_NEON" NOTES: NEON DOES NOT EXISTS ON FCC
#./create_test.sh test_3_sve "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"

# export OMP_SCHEDULE="static" 
# ./create_test.sh test_4_sve_static " -DHPCG_USE_SVE -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS"
# export OMP_SCHEDULE="static,18" 
# ./create_test.sh test_5_sve_static18 " -DHPCG_USE_SVE -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS"
# export OMP_SCHEDULE="static,36" 
# ./create_test.sh test_6_sve_static36 " -DHPCG_USE_SVE -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS"
# export OMP_SCHEDULE="static,72" 
# ./create_test.sh test_7_sve_static72 " -DHPCG_USE_SVE -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS"

#./create_test.sh test_8_sve_160 " -DHPCG_USE_SVE -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS" 160
#./create_test.sh test_9_sve_192 " -DHPCG_USE_SVE -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS" 192
#./create_test.sh test_9_sve_128 " -DHPCG_USE_SVE -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS" 128

#./create_test.sh test_10_noopt_144_isolate_mem " -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS"
#./create_test.sh test_11_SVE_144_isolate_mem "-DHPCG_USE_SVE  -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS"
#./create_test.sh test_12_SVE_144_nounroll_and_jam "-DHPCG_USE_SVE  -DHPCG_MAN_OPT_SCHEDULE_ON -DENABLE_MG_COUNTERS"

#./create_test.sh test_3b_sve "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
# ./create_test.sh test_3e_sve "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_4_sve_2unroll "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_5_sve_4unroll "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_6_sve_2unroll "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_7_sve_4unroll2 "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
#./create_test.sh test_8_sve_4unroll3 "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"     #nclang flags
#./create_test.sh test_9_sve_4unroll4 "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"      #current flags
#./create_test.sh test_10_sve_4unroll4 "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"      #nclang flags - no barrier
#./create_test.sh test_11_sve_4unroll4 "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"      #nclang flags - no barrier
export OMP_WAIT_POLICY=active
./create_test.sh test_12__sve_4unroll4_activewait "-DENABLE_MG_COUNTERS -DHPCG_USE_SVE"
