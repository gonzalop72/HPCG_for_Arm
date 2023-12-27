#!/bin/bash -l
# export OMP_SCHEDULE="static,160" 
# ./create_test.sh test_NEW_B2_unroll4_static_160 "-DHPCG_MAN_OPT_SPMV_UNROLL -DSPMV_4_UNROLL -Khpctag -Kzfill -DHPCG_MAN_OPT_SCHEDULE_ON" 160
./create_test.sh test_multinode_176 "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_528" 176
./create_test.sh test_multinode_160 "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_720" 160
