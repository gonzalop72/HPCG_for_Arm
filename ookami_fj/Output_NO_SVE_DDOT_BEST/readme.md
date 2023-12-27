TEST:
MOTIVATION: maximise DDOT Performance for FUJITSU

HPCG_OPTS     = -DHPCG_CONTIGUOUS_ARRAYS -DHPCG_NO_MPI
CXX          = FCC
CXXFLAGS     = $(HPCG_DEFS) -Nclang -O3 -ffast-math -Kfast -Kopenmp -funroll-loops -std=c++11 -ffp-contract=fast -march=armv8.2-a+sve -Koptmsg=2 -Nlst=t


