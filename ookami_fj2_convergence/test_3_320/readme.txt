The idea of this test is to use the program compiled on: 
/lustre/home/gapinzon/arm_code/HPCG_for_Arm/ookami_fj2_mpi_2/test_multinode_160

to do the 2-node comparison between the reference implementation (80 "pure MPI" 1-core process) vs this optimized implementation (8 10-thread processes).
nx=ny=nz=160
The test is set for 30 minutes.