#!/bin/bash -l
module load gcc/12.2.0
module load likwid/5.2.2
#5) arm-modules/22.1   6) binutils/11.2.0   7) acfl/22.1  

export ARMPL_DIR=/lustre/software/arm/22.1/armpl-22.1.0_AArch64_RHEL-8_gcc_aarch64-linux
export ARMPL_INC=$ARMPL_DIR/include
export ARMPL_LIB=$ARMPL_DIR/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARMPL_DIR/lib
