#pragma once

#include <getopt.h>
#include <string>
#include <cstdlib>

//////////////////////////////
//////////////////////////////

#define DEBUG false
#define RESET true
// Paths are relative to the host executable!
//#define DEFAULT_MTX_FILE "../../data/matrices_for_testing/small_test_real.mtx"
//#define DEFAULT_MTX_FILE "../../data/matrices_for_testing/documents/matr_1000_docs_max_degree_20.mtx"
//#define DEFAULT_MTX_FILE "../../data/matrices_for_testing/matrices_512/new_mat_size_1000.txt"
#define DEFAULT_MTX_FILE "../../data/matrices_for_testing/matrices_small/matrix_1000_512_20_gamma.mtx"
//#define DEFAULT_MTX_FILE "../../data/matrices_for_testing/matrices_small/matrix_100000_100000_20_uniform.mtx"
//#define DEFAULT_MTX_FILE "../../data/matrices_for_testing/dense_16_16.mtx"

#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 64
#define DEFAULT_GPU_IMPL 0
#define DEFAULT_USE_HALF_PRECISION_GPU false

#define DEFAULT_NUM_TESTS 3

#define DEFAULT_TOP_K 20

#define XCLBIN "../approximate_spmv.xclbin"

//////////////////////////////
//////////////////////////////

enum GPU_IMPL { CSR = 0, CSR_LIGHTSPMV = 1, COO = 2 };

struct Options {

    // Input-specific options;
    std::string matrix_path = DEFAULT_MTX_FILE;
    bool use_sample_matrix = false;
    bool reset = RESET;
    // Testing options;
    uint num_tests = DEFAULT_NUM_TESTS;
    int debug = DEBUG;
    bool ignore_matrix_values = false;

    int top_k_value = DEFAULT_TOP_K;

    // FPGA-specific options;
    std::string xclbin_path = XCLBIN;
    // GPU-specific options;
    GPU_IMPL gpu_impl = GPU_IMPL(DEFAULT_GPU_IMPL);
    bool use_half_precision_gpu = DEFAULT_USE_HALF_PRECISION_GPU;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[]) {
        // m: path to the directory that stores the input matrix, stored as MTX
        // s: use a small example matrix instead of the input files
        // d: if present, print all debug information, else a single summary line at the end
        // t: if present, repeat the computation the specified number of times
    	// x: xclbin path
    	// v: if present, ignore values in the matrix and set all of them to 1; used to load graph topologies
        int opt;
        static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                               {"use_sample_matrix", no_argument, 0, 's'},
											   {"no_reset", no_argument, 0, 'r'},
                                               {"matrix_path", required_argument, 0, 'm'},
                                               {"num_tests", required_argument, 0, 't'},
                                               {"xclbin", required_argument, 0, 'x'},
											   {"ignore_matrix_values", no_argument, 0, 'v'},
											   {"k", required_argument, 0, 'k'},
                                               {"block_size_1d", required_argument, 0, 'b'},
                                               {"block_size_2d", required_argument, 0, 'c'},
                                               {"num_blocks", required_argument, 0, 'g'},
                                               {"gpu_impl", required_argument, 0, 'i'}, // Used to choose between cuSPARSE and LightSpMV in CSR format, unnecessary for COO;
                                               {"half_precision_gpu", no_argument, 0, 'a'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "dm:st:x:vk:rb:c:g:i:a", long_options, &option_index)) != EOF) {
            switch (opt) {
                case 'd':
                    debug = true;
                    break;
                case 'r':
                	reset = true;
					break;
                case 'm':
                    matrix_path = optarg;
                    break;
                case 's':
                    use_sample_matrix = true;
                    break;
                case 't':
                    num_tests = atoi(optarg);
                    break;
                case 'x':
                    xclbin_path = optarg;
                    break;
                case 'v':
                	ignore_matrix_values = true;
					break;
                case 'k':
					top_k_value = atoi(optarg);
					break;
                case 'b':
                    block_size_1d = atoi(optarg);
                    break;
                case 'c':
                    block_size_2d = atoi(optarg);
                    break;
                case 'g':
                    num_blocks = atoi(optarg);
                    break;
                case 'i':
                    gpu_impl = GPU_IMPL(atoi(optarg));
                    break;
                case 'a':
                    use_half_precision_gpu = true;
                    break;
                default:
                    break;
            }
        }
    }
};
