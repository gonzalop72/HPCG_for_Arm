#!/bin/bash -l
#./create_test.sh test1b_basic_optimizations ""
#./create_test.sh test1_empty "-Khpctag"
#./create_test.sh test2_nosve ""
#./create_test.sh test2_sve "-DHPCG_USE_SVE"
#./create_test.sh test3_neon "-DHPCG_USE_NEON"
#./create_test.sh test4_armpl "-DHPCG_USE_ARMPL_SPMV"
#./create_test.sh test5_unroll2 "-DHPCG_MAN_OPT_SPMV_UNROLL"
#./create_test.sh test6_unroll2_2 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_2_UNROLL"
#./create_test.sh test7_unroll4_2 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL"
#./create_test.sh test8_unroll2_3 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_2_UNROLL  -Khpctag"
#./create_test.sh test9_unroll4_3 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL  -Khpctag"
#./create_test.sh test10_unroll2_4 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_2_UNROLL  -Khpctag"
#./create_test.sh test10b_unroll2_4 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_2_UNROLL  -Khpctag"
#./create_test.sh test11_unroll4_2 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill"
#./create_test.sh test10c_unroll2_5 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_2_UNROLL -Khpctag -Kzfill "
#./create_test.sh test12_unroll4_2_static "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "

#./create_test.sh test_NEW_1_static "-Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
#./create_test.sh test_NEW_unroll2_static "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_2_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
#./create_test.sh test_NEW_unroll4_static "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
export OMP_SCHEDULE="static,18" 
./create_test.sh test_NEW_4_unroll4_static_18 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
export OMP_SCHEDULE="static,36" 
./create_test.sh test_NEW_5_unroll4_static_36 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
export OMP_SCHEDULE="static,72" 
./create_test.sh test_NEW_6_unroll4_static_72 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON "
export OMP_SCHEDULE="static,144" 
./create_test.sh test_NEW_7_unroll4_static_144 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON"
export OMP_SCHEDULE="static"
./create_test.sh test_NEW_8_unroll4_static_144 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON -DHPCG_USE_NEON"
./create_test.sh test_NEW_9_unroll4_static_144 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON -DHPCG_USE_SVE"
export OMP_SCHEDULE="static" 
./create_test.sh test_NEW_B1_unroll4_static "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON" 160
export OMP_SCHEDULE="static,160" 
./create_test.sh test_NEW_B2_unroll4_static_160 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON" 160

export OMP_WAIT_POLICY=active 
OMP_SCHEDULE="static,640" ./create_test.sh test_004_unroll4_sch "-DUNROLLING_4_A" 176
OMP_SCHEDULE="static,640" ./create_test.sh test_004_unroll4_man "-DUNROLLING_4_B" 176