#!/bin/bash -l
export OMP_WAIT_POLICY=active 

#./create_test.sh test_001_D_full "-DUNROLLING_6_B"
#./create_test.sh test_001B_D_full "-DUNROLLING_6_B" 160
#./create_test.sh test_001C_D_full "-DUNROLLING_6_B" 176
#symgs manual scheduling
#./create_test.sh test_001_D_man "-DUNROLLING_6_C" 144 0
#./create_test.sh test_001B_D_man "-DUNROLLING_6_C" 160 0
#./create_test.sh test_001C_D_man "-DUNROLLING_6_C" 176 0

######## test E ############
#./create_test.sh test_001A_E_ref "-DREF_UNROLLING_4"
#./create_test.sh test_001B_E_ref "-DREF_UNROLLING_4" 160
#./create_test.sh test_001C_E_ref "-DREF_UNROLLING_4" 176

#./create_test.sh test_001A_E_full "-DUNROLLING_4_A"
#./create_test.sh test_001B_E_full "-DUNROLLING_4_A" 160
#./create_test.sh test_001C_E_full "-DUNROLLING_4_A" 176

#symgs manual scheduling
#./create_test.sh test_001A_E_man "-DUNROLLING_4_B" 144 0
#./create_test.sh test_001B_E_man "-DUNROLLING_4_B" 160 0
#./create_test.sh test_001C_E_man "-DUNROLLING_4_B" 176 0

######## test F ############

#manual scheduling with fixed static,720
#./create_test.sh test_001A_F_man "-DUNROLLING_4_B" 144 0
#./create_test.sh test_001B_F_man "-DUNROLLING_4_B" 160 0
#./create_test.sh test_001C_F_man "-DUNROLLING_4_B" 176 0
#./create_test.sh test_001D_F_man "-DUNROLLING_4_B" 192 0

#manual scheduling with fixed static,720 and ddot 4-unrolled intrinsics
#./create_test.sh test_001A_G_man "-DUNROLLING_4_B" 144 0
#./create_test.sh test_001B_G_man "-DUNROLLING_4_B" 160 0
#./create_test.sh test_001C_G_man "-DUNROLLING_4_B" 176 0
#./create_test.sh test_001D_G_man "-DUNROLLING_4_B" 192 0

#manual scheduling with fixed static,720 and ddot 2-unrolled
#./create_test.sh test_001A_H_man "-DUNROLLING_4_B" 144 0
#./create_test.sh test_001B_H_man "-DUNROLLING_4_B" 160 0
#./create_test.sh test_001C_H_man "-DUNROLLING_4_B" 176 0
#./create_test.sh test_001D_H_man "-DUNROLLING_4_B" 192 0

#manual scheduling with fixed static,720 and ddot 4-unrolled intrinsics (again)
#./create_test.sh test_001A_I_man "-DUNROLLING_4_B" 144 0
#./create_test.sh test_001B_I_man "-DUNROLLING_4_B" 160 0
#./create_test.sh test_001C_I_man "-DUNROLLING_4_B" 176 0
#./create_test.sh test_001D_I_man "-DUNROLLING_4_B" 192 0

#manual scheduling with fixed static,720 and ddot 4-unrolled intrinsics (again)
#./create_test.sh test_singlenode_144_man "-DUNROLLING_4_B" 144 720
#./create_test.sh test_singlenode_160_man "-DUNROLLING_4_B" 160 720
#./create_test.sh test_singlenode_176_man "-DUNROLLING_4_B" 176 720
#./create_test.sh test_singlenode_176_man_2 "-DUNROLLING_4_B" 176 528

#./create_test.sh test_singlenode_176_man_B2 "-DUNROLLING_4_B" 176 528
#./create_test.sh test_singlenode_176_man_B "-DUNROLLING_4_B" 176 720
#./create_test.sh test_singlenode_144_man_B "-DUNROLLING_4_B" 144 720
#./create_test.sh test_singlenode_160_man_B "-DUNROLLING_4_B" 160 720

#thesis doc result
#./create_test.sh test_singlenode_176_man_30min "-DUNROLLING_4_B" 176 528

#./create_test.sh test_singlenode_144_man_30min_fixed720 "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_720" 144
#./create_test.sh test_singlenode_160_man_30min_fixed720 "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_720" 160
#./create_test.sh test_singlenode_176_man_30min_fixed720 "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_720" 176
#./create_test.sh test_singlenode_176_man_30min_fixed528 "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_528" 176

#./create_test.sh test_singlenode_176_xx "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_528" 176
#./create_test.sh test_singlenode_192_xx "-DUNROLLING_4_B -DHPCG_MAN_SPVM_SCHEDULE_720" 192

#current presentation values are
#test_001*_E_***  :

######## test J ############
#like test E but expanding the search range
./create_test.sh test_001A_J_ref "-DREF_UNROLLING_4"
./create_test.sh test_001B_J_ref "-DREF_UNROLLING_4" 160
./create_test.sh test_001C_J_ref "-DREF_UNROLLING_4" 176

./create_test.sh test_001A_J_full "-DUNROLLING_4_A"
./create_test.sh test_001B_J_full "-DUNROLLING_4_A" 160
./create_test.sh test_001C_J_full "-DUNROLLING_4_A" 176

#symgs manual scheduling
./create_test.sh test_001A_J_man "-DUNROLLING_4_B" 144 0
./create_test.sh test_001B_J_man "-DUNROLLING_4_B" 160 0
./create_test.sh test_001C_J_man "-DUNROLLING_4_B" 176 0