TEST: 

COMPILER: GCC 11.2

ADDITIONAL LIBRARIES: 		

CONFIG FILE: arm_code/v2022/HPCG_for_Arm/Fritz_OMP_GCC/setup/Make.Fritz_OMP_GCC
CXX          = g++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -ftree-vectorize -ftree-vectorizer-verbose=0 -fopenmp -march=native
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_MAN_OPT_DDOT