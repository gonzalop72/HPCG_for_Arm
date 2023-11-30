TEST:

COMPILER: ARM 22.1
SVE ENABLED

ADDITIONAL LIBRARIES: 		    ARMPL
USING MANUAL OPTIMIZATION: 	    -DHPCG_MAN_OPT_DDOT 

Iteration 1 (NO OPT):
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE 

Iteration 2 (DDOT OPT):
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_MAN_OPT_DDOT 

Iteration 3 (DDOT OPT + ARMPL SPMV)
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_MAN_OPT_DDOT -DHPCG_USE_ARMPL_SPMV

Iteration 4 (SPMV UNROLL OPT)
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_MAN_OPT_SPMV_UNROLL

Iteration 5 (ARMPL SPMV + ARMPL DDOT + ARMPL WAXBY)
CXX          = armclang++
CXXFLAGS     = $(HPCG_DEFS) -Ofast -ffast-math -fvectorize -fopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve
HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI -DHPCG_USE_SVE -DHPCG_USE_ARMPL_SPMV -DHPCG_USE_DDOT_ARMPL -DHPCG_USE_WAXPBY_ARMPL
