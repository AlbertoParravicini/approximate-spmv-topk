import subprocess
import time
import os
from datetime import datetime

date = datetime.now()

##########################################################
# These are configuration parameters that you can change #
##########################################################

TESTS = ["fpga"]
DEBUG = False
DATE = "2020_11_12"  # Output date;
MATRIX_SIZES = [10_000, 100_000]
MATRIX_SIZES = [5_000_000, 10_000_000, 15_000_000]
MATRIX_COLS = [512, 1024]
MATRIX_DIST = ["uniform", "gamma"]
MATRIX_NNZ = [20, 40]
K = 100
NITER = 30

# This is the folder where your matrices are stored, as .mtx files;
MATRIX_FOLDER = "/path/to/matrices"
# Here you can specify some additional matrices that don't follow the standard naming pattern of "matrix_{rows}_{columns}_{nnz}_{distribution}.mtx"
ADDITIONAL_MATRICES = [{
    "name": "/path/to/matrices/glove_sparse/glove_2.2M_y300_0indexed.mtx",
    "s": 2196017,
    "c": 300,
    "d": "glove",
    "n": 54900425
    }]

OUT_FOLDER = f"data/results/{date.strftime('%Y_%m_%d_%H_%M_%S')}"
ZERO_INDEXED = True
SKIP_STANDARD_MATRICES = False
SKIP_ADDITIONAL_MATRICES = False

# FPGA parameters;
FPGA_BUILD_DIR = "/path/to/build/approximate-spmv-topk-builds"  # Path to the FPGA build directory;
# List of FPGA builds to be tested, they all have name equal to "{fpga_platform}_{num_cores}_{frequency}_{precision}";
FPGA_BUILDS = [
	"xilinx_u280_xdma_201920_3_24core_219mhz_32bit",
	"xilinx_u280_xdma_201920_3_24core_235mhz_26bit",
	"xilinx_u280_xdma_201920_3_32core_172mhz_21bit",
	"xilinx_u280_xdma_201920_3_24core_229mhz_float"
]
# Name of FPGA xclbin, as specified in the Makefile;
FPGA_XCLBIN = "spmv_bscsr_top_k_main.xclbin"
# Name of the FPGA host executable, as specified in the Makefile;
FPGA_MAIN = "spmv_coo_hbm_topk_multicore_mega_main"

# GPU parameters, which GPU configurations will be tested;
GPU_VERSIONS = [0, 1, 2]
GPU_USE_HALF = [False, True]
GPU_VERSION_MAP = {0: "csr", 1: "csr", 2: "coo"}

############################################################
# Start of script, no need to configure anything from here #
############################################################

FPGA_CMD = "{}/{}/{} -x {}/{}/{} -m {} -t {} -k {} {} | tee {}"                            
GPU_CMD = "src/gpu/bin/approximate-spmv-gpu-{}-topk {} -t {} -m {} -k {} -i {} {} -r | tee {}"
CPU_CMD = "python3 test_cpu.py {} -t {} {} -i {} -k {} -o {}"

def test(curr_test, t, s, c, d, n, input_matrix):
    if t == "fpga":
        for j in range(len(FPGA_BUILDS)):
            curr_test += 1 
            _, _, _, _, _, cores, clock, bits = FPGA_BUILDS[j].split("_")
            output_file = os.path.join(OUT_FOLDER, f"{t}_{s}_{c}_{d}_{n}_{bits}_{cores}_{clock}_{K}_{NITER}.csv")
            cmd = FPGA_CMD.format(FPGA_BUILD_DIR, FPGA_BUILDS[j], FPGA_MAIN, FPGA_BUILD_DIR, FPGA_BUILDS[j], FPGA_XCLBIN, input_matrix, NITER, K, "-d" if DEBUG else "", output_file)
            print(f"running {curr_test}/{num_tests}={cmd}")
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif t == "gpu":
        for v in GPU_VERSIONS:
            for h in GPU_USE_HALF:
                curr_test += 1 
                output_file = os.path.join(OUT_FOLDER, f"{t}_{s}_{c}_{d}_{n}_{v}_{h}_{K}_{NITER}.csv")
                cmd = GPU_CMD.format(GPU_VERSION_MAP[v], "-d" if DEBUG else "", NITER, input_matrix, K, v, "-a" if h else "", output_file)
                print(f"running {curr_test}/{num_tests}={cmd}")
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif t == "cpu":
        curr_test += 1 
        output_file = os.path.join(OUT_FOLDER, f"{t}_{s}_{c}_{d}_{n}_{K}_{NITER}.csv")
        cmd = CPU_CMD.format("-d" if DEBUG else "", NITER, "-z" if ZERO_INDEXED else "", input_matrix, K, output_file)
        print(f"running {curr_test}/{num_tests}={cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return curr_test, result
        
#############################
#############################
        
if __name__ == "__main__":

    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    num_tests = (len(ADDITIONAL_MATRICES) * (not SKIP_ADDITIONAL_MATRICES) + (len(MATRIX_SIZES) * len(MATRIX_COLS) * len(MATRIX_DIST) * len(MATRIX_NNZ)) * (not SKIP_STANDARD_MATRICES)) * ((len(FPGA_BUILDS) * ("fpga" in TESTS)) + (len(GPU_VERSIONS) * len(GPU_USE_HALF) * ("gpu" in TESTS)) + ("cpu" in TESTS))
    curr_test = 0
    for t in TESTS:
        if not SKIP_STANDARD_MATRICES:
            for s in MATRIX_SIZES:
                for c in MATRIX_COLS:
                    for d in MATRIX_DIST:
                        for n in MATRIX_NNZ:
                            input_matrix = os.path.join(MATRIX_FOLDER, f"matrix_{s}_{c}_{n}_{d}.mtx")
                            curr_test, result = test(curr_test, t, s, c, d, n, input_matrix)
        if not SKIP_ADDITIONAL_MATRICES:
            for a in ADDITIONAL_MATRICES:
                curr_test, result = test(curr_test, t, a["s"], a["c"], a["d"], a["n"], a["name"])
