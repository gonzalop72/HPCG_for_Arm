#/usr/bin/sh -l

cd test_001A_E_ref
sbatch run_test3.sh
cd -
cd test_001B_E_ref
sbatch run_test3.sh  160
cd -
cd test_001C_E_ref
sbatch run_test3.sh 176
cd -

cd test_001A_E_full
sbatch run_test3.sh
cd -
cd test_001B_E_full
sbatch run_test3.sh 160
cd -
cd test_001C_E_full
sbatch run_test3.sh 176
cd -

#symgs manual scheduling
cd test_001A_E_man
sbatch run_test3.sh 144 0
cd -
cd test_001B_E_man
sbatch run_test3.sh 160 0
cd -
cd test_001C_E_man
sbatch run_test3.sh 176 0
cd -