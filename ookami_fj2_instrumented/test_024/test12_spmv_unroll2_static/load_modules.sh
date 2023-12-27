#!/bin/bash -l

module load fujitsu/compiler/4.7  
module load likwid/5.2.2
module load arm-modules/22.1

#NOTES: USING ARMPL libraries from ARM 22.1 compilation
export ARMPL_DIR=/lustre/software/arm/22.1/armpl-22.1.0_AArch64_RHEL-8_arm-linux-compiler_aarch64-linux
export ARMPL_INC=$ARMPL_DIR/include
export ARMPL_LIB=$ARMPL_DIR/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARMPL_DIR/lib
