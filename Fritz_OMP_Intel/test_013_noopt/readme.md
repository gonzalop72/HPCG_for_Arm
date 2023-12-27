TEST: 

COMPILER: Intel 2021.4.0

ADDITIONAL LIBRARIES: 		

CONFIG FILE: arm_code/v2022/HPCG_for_Arm/Fritz_OMP_Intel/setup/Make.Fritz_OMP_Intel
CXX          = icpc
CXXFLAGS     = $(HPCG_DEFS) -O3 -qopenmp -std=c++11 -xCORE-AVX512 -qopt-zmm-usage=high
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI

NO SCHEDULE: unset OMP_SCHEDULE -> default = static
