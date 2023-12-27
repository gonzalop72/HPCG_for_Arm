ComputeDotProduct.asm1.out:
GCC 12.2.0 - MANUAL UNROLLING - 4

g++ -c -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_MAN_OPT_DDOT -DHPCG_USE_ARMPL_SPMV -I./src -I./src/QPACE4_OMP_GCC+SVE  -I/lustre/software/arm/22.1/armpl-22.1.0_AArch64_RHEL-8_gcc_aarch64-linux/include  -Ofast -ffast-math -ftree-vectorize -ftree-vectorizer-verbose=0 -fopenmp -funroll-loops -std=c++11 -march=armv8.2-a+sve -I../src 
-S ../src/ComputeDotProduct.cpp -o ComputeDotProduct.asm1.out
