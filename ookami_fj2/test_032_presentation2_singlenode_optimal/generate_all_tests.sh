#!/bin/bash -l
export OMP_WAIT_POLICY=active 
#OMP_SCHEDULE="static,720" ./create_test.sh test_001_full_144_720 "" 144
#OMP_SCHEDULE="static,720" ./create_test.sh test_001_full_160_720 "" 160
#OMP_SCHEDULE="static,144" ./create_test.sh test_001_full_144_144 "" 144
#OMP_SCHEDULE="static,560" ./create_test.sh test_001_full_160_560 "" 160

#OMP_SCHEDULE="static,720" ./create_test.sh test_002_full_144_720 "" 144
#OMP_SCHEDULE="static,720" ./create_test.sh test_002_full_160_720 "" 160
#OMP_SCHEDULE="static,144" ./create_test.sh test_002_full_144_144 "" 144
#OMP_SCHEDULE="static,560" ./create_test.sh test_003_full_160_560 "" 160
OMP_SCHEDULE="static,320" ./create_test.sh test_003_full_176_320 "" 176
OMP_SCHEDULE="static,640" ./create_test.sh test_004_unroll4_sch "-DUNROLLING_4_A" 176
OMP_SCHEDULE="static,640" ./create_test.sh test_004_unroll4_man "-DUNROLLING_4_B" 176
OMP_SCHEDULE="static,640" ./create_test.sh test_004_unroll4_sch "-DUNROLLING_6_A" 176

