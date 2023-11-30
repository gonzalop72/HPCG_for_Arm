#!/bin/bash -l
export OMP_WAIT_POLICY=active 
#OMP_SCHEDULE="static,720" ./create_test.sh test_001_full_144_720 "" 144
#OMP_SCHEDULE="static,720" ./create_test.sh test_001_full_160_720 "" 160
#OMP_SCHEDULE="static,144" ./create_test.sh test_001_full_144_144 "" 144
#OMP_SCHEDULE="static,560" ./create_test.sh test_001_full_160_560 "" 160

OMP_SCHEDULE="static,720" ./create_test.sh test_002_full_144_720 "" 144
OMP_SCHEDULE="static,720" ./create_test.sh test_002_full_160_720 "" 160
OMP_SCHEDULE="static,144" ./create_test.sh test_002_full_144_144 "" 144
OMP_SCHEDULE="static,560" ./create_test.sh test_002_full_160_560 "" 160
