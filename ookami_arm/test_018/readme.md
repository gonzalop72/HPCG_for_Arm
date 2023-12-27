TEST 018
MOTIVATION: TEST LARGE MEMORY MODEL - FURTHER IMPROVEMENTS

COMPILER: ARM 22.1
SVE ENABLED

ADDITIONAL LIBRARIES: 		    ARMPL


Iteration 1 : ALL ARMPL OPTIMIZATIONS - LARGE MEMORY MODEL
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve -mcmodel=large
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_USE_ARMPL_SPMV -DHPCG_USE_DDOT_ARMPL -DHPCG_USE_WAXPBY_ARMPL

Iteration 2 : ALL ARMPL OPTIMIZATIONS - NO LARGE MEMORY MODEL
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE

Iteration 3 : MANUAL SPMV OPT - ARMPL OPTIMIZATIONS ON DDOT ONLY - NO LARGE MEMORY MODEL
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_USE_DDOT_ARMPL

Iteration 4 : ARMPL OPTIMIZATIONS ON DDOT AND SPMV - NO LARGE MEMORY MODEL - SEEMS OPTIMAL
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_USE_DDOT_ARMPL -DHPCG_USE_ARMPL_SPMV