#######################################################################################################################################
#
#	Basic Makefile for Vitis 2019.2
#	Lorenzo Di Tucci, Emanuele Del Sozzo, Alberto Parravicini
#	{lorenzo.ditucci, emanuele.delsozzo, albetro.parravicini}@polimi.it
#	Usage make [emulation | build | clean | clean_sw_emu | clean_hw_emu | clean_hw | cleanall] TARGET=<sw_emu | hw_emu | hw>
#
#
#######################################################################################################################################

XOCC=v++
CC=g++

#############################
# Define files to compile ###
#############################

# Host code
HOST_SRC=./src/fpga/src/host_spmv_bscsr.cpp
HOST_HEADER_DIRS=.

# Host header files (optional, used to check if rebuild is required);
UTILS_DIR=./src/common/utils
FPGA_DIR=./src/fpga/src
HOST_HEADERS=./src/common/csc_matrix/csc_matrix.hpp $(UTILS_DIR)/evaluation_utils.hpp $(UTILS_DIR)/mmio.hpp $(UTILS_DIR)/options.hpp $(UTILS_DIR)/utils.hpp $(FPGA_DIR)/aligned_allocator.h $(FPGA_DIR)/gold_algorithms/gold_algorithms.hpp $(FPGA_DIR)/opencl_utils.hpp $(FPGA_DIR)/ip/coo_fpga.hpp

# Name of host executable
HOST_EXE=spmv_coo_hbm_topk_multicore_mega_main

# Kernel
KERNEL_DIR=./src/fpga/src/ip/spmv
KERNEL_SRC=$(KERNEL_DIR)/spmv_bscsr_top_k_multicore.cpp 
KERNEL_HEADER_DIRS=./src/fpga/src/ip/spmv
KERNEL_FLAGS=
# Name of the xclbin;
KERNEL_EXE=spmv_bscsr_top_k_main
# Name of the main kernel function to build;
KERNEL_NAME=spmv_bscsr_top_k_main

#############################
# Define FPGA & host flags  #
#############################

# Target clock of the FPGA, in MHz;
TARGET_CLOCK=450
# Port width, in bit, of the kernel;
PORT_WIDTH=512

# Device code for Alveo U200;
ALVEO_U280=xilinx_u280_xdma_201920_3
ALVEO_U280_DEVICE="\"xilinx_u280_xdma_201920_3"\"
TARGET_DEVICE=$(ALVEO_U280)

# Flags to provide to xocc, specify here associations between memory bundles and physical memory banks.
# Documentation: https://www.xilinx.com/html_docs/xilinx2019_1/sdaccel_doc/wrj1504034328013.html
KERNEL_LDCLFLAGS=--xp param:compiler.preserveHlsOutput=1 \
	--nk $(KERNEL_NAME):8:$(KERNEL_NAME)_1.$(KERNEL_NAME)_2.$(KERNEL_NAME)_3.$(KERNEL_NAME)_4.$(KERNEL_NAME)_5.$(KERNEL_NAME)_6.$(KERNEL_NAME)_7.$(KERNEL_NAME)_8\
	--connectivity.slr $(KERNEL_NAME)_1:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_2:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_3:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_4:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_5:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_6:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_7:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_8:SLR1 \
	--sp $(KERNEL_NAME)_1.m_axi_gmem0:HBM[0] \
	--sp $(KERNEL_NAME)_1.m_axi_gmem1:HBM[1] \
	--sp $(KERNEL_NAME)_1.m_axi_gmem2:HBM[2] \
	--sp $(KERNEL_NAME)_1.m_axi_gmem3:HBM[3] \
	--sp $(KERNEL_NAME)_2.m_axi_gmem0:HBM[4] \
	--sp $(KERNEL_NAME)_2.m_axi_gmem1:HBM[5] \
	--sp $(KERNEL_NAME)_2.m_axi_gmem2:HBM[6] \
	--sp $(KERNEL_NAME)_2.m_axi_gmem3:HBM[7] \
	--sp $(KERNEL_NAME)_3.m_axi_gmem0:HBM[8] \
 	--sp $(KERNEL_NAME)_3.m_axi_gmem1:HBM[9] \
	--sp $(KERNEL_NAME)_3.m_axi_gmem2:HBM[10] \
	--sp $(KERNEL_NAME)_3.m_axi_gmem3:HBM[11] \
	--sp $(KERNEL_NAME)_4.m_axi_gmem0:HBM[12] \
	--sp $(KERNEL_NAME)_4.m_axi_gmem1:HBM[13] \
	--sp $(KERNEL_NAME)_4.m_axi_gmem2:HBM[14] \
	--sp $(KERNEL_NAME)_4.m_axi_gmem3:HBM[15] \
	--sp $(KERNEL_NAME)_5.m_axi_gmem0:HBM[16] \
	--sp $(KERNEL_NAME)_5.m_axi_gmem1:HBM[17] \
	--sp $(KERNEL_NAME)_5.m_axi_gmem2:HBM[18] \
	--sp $(KERNEL_NAME)_5.m_axi_gmem3:HBM[19] \
	--sp $(KERNEL_NAME)_6.m_axi_gmem0:HBM[20] \
 	--sp $(KERNEL_NAME)_6.m_axi_gmem1:HBM[21] \
	--sp $(KERNEL_NAME)_6.m_axi_gmem2:HBM[22] \
	--sp $(KERNEL_NAME)_6.m_axi_gmem3:HBM[23] \
	--sp $(KERNEL_NAME)_7.m_axi_gmem0:HBM[24] \
	--sp $(KERNEL_NAME)_7.m_axi_gmem1:HBM[25] \
	--sp $(KERNEL_NAME)_7.m_axi_gmem2:HBM[26] \
	--sp $(KERNEL_NAME)_7.m_axi_gmem3:HBM[27] \
	--sp $(KERNEL_NAME)_8.m_axi_gmem0:HBM[28] \
 	--sp $(KERNEL_NAME)_8.m_axi_gmem1:HBM[29] \
	--sp $(KERNEL_NAME)_8.m_axi_gmem2:HBM[30] \
	--sp $(KERNEL_NAME)_8.m_axi_gmem3:HBM[31] \

KERNEL_ADDITIONAL_FLAGS=--kernel_frequency $(TARGET_CLOCK) -j 40 -O3

#  Specify host compile flags and linker;
HOST_INCLUDES= -I${XILINX_XRT}/include -I${XILINX_VITIS}/include
HOST_CFLAGS=$(HOST_INCLUDES) -D TARGET_DEVICE=$(ALVEO_U280_DEVICE) -g -D C_KERNEL -O3 -std=c++1y -pthread -lrt -lstdc++ 
HOST_LFLAGS=-L${XILINX_XRT}/lib -lxilinxopencl -lOpenCL

##########################################
# No need to modify starting from here ###
##########################################

#############################
# Define compilation type ###
#############################

# TARGET for compilation [sw_emu | hw_emu | hw | host]
TARGET=none
REPORT_FLAG=n
REPORT=
ifeq (${TARGET}, sw_emu)
$(info software emulation)
TARGET=sw_emu
ifeq (${REPORT_FLAG}, y)
$(info creating REPORT for software emulation set to true. This is going to take longer as it will synthesize the kernel)
REPORT=--report estimate
else
$(info I am not creating a REPORT for software emulation, set REPORT_FLAG=y if you want it)
REPORT=
endif
else ifeq (${TARGET}, hw_emu)
$(info hardware emulation)
TARGET=hw_emu
REPORT=--report estimate
else ifeq (${TARGET}, hw)
$(info system build)
TARGET=hw
REPORT=--report system
else
$(info no TARGET selected)
endif

PERIOD:= :
UNDERSCORE:= _
DEST_DIR=build/$(TARGET)/$(subst $(PERIOD),$(UNDERSCORE),$(TARGET_DEVICE))

#############################
# Define targets ############
#############################

clean:
	rm -rf .Xil emconfig.json 

clean_sw_emu: clean
	rm -rf sw_emu
clean_hw_emu: clean
	rm -rf hw_emu
clean_hw: clean
	rm -rf hw

cleanall: clean_sw_emu clean_hw_emu clean_hw
	rm -rf _xocc_* xcl_design_wrapper_*

check_TARGET:
ifeq (${TARGET}, none)
	$(error Target can not be set to none)
endif

host:  check_TARGET $(HOST_SRC) $(HOST_HEADERS)
	mkdir -p $(DEST_DIR)
	$(CC) $(HOST_SRC) $(HOST_CFLAGS) $(HOST_LFLAGS) -o $(DEST_DIR)/$(HOST_EXE)

xo:	check_TARGET
	mkdir -p $(DEST_DIR)
	$(XOCC) --platform $(TARGET_DEVICE) --target $(TARGET) --compile --include $(KERNEL_HEADER_DIRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(KERNEL_SRC) $(KERNEL_LDCLFLAGS) $(KERNEL_FLAGS) $(KERNEL_ADDITIONAL_FLAGS) --output $(DEST_DIR)/$(KERNEL_EXE).xo

xclbin:  check_TARGET xo
	$(XOCC) --platform $(TARGET_DEVICE) --target $(TARGET) --link --include $(KERNEL_HEADER_DIRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(DEST_DIR)/$(KERNEL_EXE).xo $(KERNEL_LDCLFLAGS) $(KERNEL_FLAGS) $(KERNEL_ADDITIONAL_FLAGS) --output $(DEST_DIR)/$(KERNEL_EXE).xclbin

emulation:  host xclbin
	export XCL_EMULATION_MODE=$(TARGET)
	emconfigutil --platform $(TARGET_DEVICE) --nd 1
	./$(DEST_DIR)/$(HOST_EXE) $(DEST_DIR)/$(KERNEL_EXE).xclbin
	$(info Remeber to export XCL_EMULATION_MODE=$(TARGET) and run emconfigutil for emulation purposes)

build:  host xclbin

run_system:  build
	./$(DEST_DIR)/$(HOST_EXE) $(DEST_DIR)/$(KERNEL_EXE)

