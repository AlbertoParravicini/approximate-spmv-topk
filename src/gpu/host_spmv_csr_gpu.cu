#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <chrono>
#include <random>
#include <cusparse.h> 

#include "../common/utils/utils.hpp"
#include "../common/utils/options.hpp"
#include "../common/utils/evaluation_utils.hpp"
#include "../fpga/src/ip/coo_matrix.hpp"
#include "../fpga/src/gold_algorithms/gold_algorithms.hpp"
#include "light_spmv.cuh"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define int_type unsigned int
#define real_type float

/////////////////////////////
/////////////////////////////

// cuSPARSE CSR SpMV example adapted from: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spmv_csr/spmv_csr_example.c

struct SpMV {

	int_type *ptr;
	int_type *idx;
	real_type *val;

	int_type *ptr_d;
	int_type *idx_d;
	real_type *val_d;
	half *val_d_half;

	int_type num_rows;
	int_type num_cols;
	int_type num_nnz;

	real_type *vec;
	real_type *vec_d;
	half *vec_d_half;

	real_type *res;
	real_type *res_d;
	half *res_d_half;

	int_type *row_counter;

	int num_blocks;
	int block_size_1d;

	cudaStream_t stream;
	cusparseHandle_t handle;

	cusparseSpMatDescr_t matrix;
    cusparseDnVecDescr_t vec_cusparse, res_cusparse;
    void *cusparse_buffer;
    size_t buffer_size;

	// SpMV cuSPARSE coefficients, no need to change them;
    real_type alpha = 1.0f;
	real_type beta = 0.0f;

	GPU_IMPL gpu_impl;
	bool use_half_precision_gpu;

	SpMV(int_type *ptr_, int_type *idx_, real_type *val_, int_type num_rows_, int_type num_cols_, int_type num_nnz_, real_type *vec_, int block_size_1d = DEFAULT_BLOCK_SIZE_1D, int debug=0, GPU_IMPL gpu_impl=GPU_IMPL(0), bool use_half_precision_gpu=DEFAULT_USE_HALF_PRECISION_GPU) :
			ptr(ptr_), idx(idx_), val(val_), num_rows(num_rows_), num_cols(num_cols_), num_nnz(num_nnz_), vec(vec_), block_size_1d(block_size_1d), gpu_impl(GPU_IMPL(gpu_impl)), use_half_precision_gpu(use_half_precision_gpu) {
		// Compute number of blocks required in the computation;
		num_blocks = ceil(num_rows / (float) block_size_1d);
		
		// Transfer data;
		setup(debug);
	}

	void setup(int debug) {

		// Setup cuSPARSE;
		cusparseStatus_t status;
	  
		status = cusparseCreate(&handle);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			if (debug) {
				std::cerr << "CUSPARSE Library initialisation failed" << std::endl;
				cusparseDestroy(handle);
				exit(1);
			}
		}

		if (debug) {
			std::cout << "Create Kernel Arguments" << std::endl;
		}
		cudaStreamCreate(&stream);
		cudaMalloc(&ptr_d, sizeof(int_type) * (num_rows + 1));
		cudaMalloc(&idx_d, sizeof(int_type) * num_nnz);
		cudaMalloc(&val_d, sizeof(real_type) * num_nnz);
		cudaMalloc(&vec_d, sizeof(real_type) * num_cols);
		cudaMalloc(&res_d, sizeof(real_type) * num_rows);
		if (use_half_precision_gpu) {
			cudaMalloc(&val_d_half, sizeof(half) * num_nnz);
			cudaMalloc(&vec_d_half, sizeof(half) * num_cols);
			cudaMalloc(&res_d_half, sizeof(half) * num_rows);
		}		
		res = (real_type*) calloc(num_rows, sizeof(real_type));

		// Transfer data from host to device;
		if (debug) {
			std::cout << "Write inputs into device memory" << std::endl;
		}
		cudaMemcpyAsync(ptr_d, ptr, sizeof(int_type) * (num_rows + 1), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(idx_d, idx, sizeof(int_type) * num_nnz, cudaMemcpyDefault, stream);
		cudaMemcpyAsync(val_d, val, sizeof(real_type) * num_nnz, cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vec_d, vec, sizeof(real_type) * num_cols, cudaMemcpyDefault, stream);
		cudaMallocManaged(&row_counter, sizeof(int_type));
		row_counter[0] = 0;
		if (use_half_precision_gpu) {
			float_to_half<<<64, 1024, 0, stream>>>(val_d, val_d_half, num_nnz);
			float_to_half<<<64, 1024, 0, stream>>>(vec_d, vec_d_half, num_cols);
		}

		// Wait for data transfer on the GPU;
		cudaDeviceSynchronize();
		if (use_half_precision_gpu) {
			cusparseCreateCsr(&matrix, num_rows, num_cols, num_nnz, ptr_d, idx_d, val_d_half, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
			cusparseCreateDnVec(&vec_cusparse, num_cols, vec_d_half, CUDA_R_16F);
		} else {
			cusparseCreateCsr(&matrix, num_rows, num_cols, num_nnz, ptr_d, idx_d, val_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
			cusparseCreateDnVec(&vec_cusparse, num_cols, vec_d, CUDA_R_32F);
		}
		
		cusparseCreateDnVec(&res_cusparse, num_rows, res_d, CUDA_R_32F);
		// Additional cuSPARSE buffer;
		cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, vec_cusparse, &beta, res_cusparse, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffer_size);
		cudaMalloc(&cusparse_buffer, buffer_size);
	}

	void operator()(int debug) {
		if (debug) {
			std::cout << "Execute the kernel" << std::endl;
		}
		auto start = clock_type::now();

		switch(gpu_impl) {
			case CSR:
				// Use cuSPARSE;
				cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, vec_cusparse, &beta, res_cusparse, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, cusparse_buffer);
				break;
			case CSR_LIGHTSPMV:
				// Use LightSpMV;
				if (use_half_precision_gpu) {
					light_spmv<<<num_blocks, block_size_1d, block_size_1d * sizeof(int_type), stream>>>(row_counter, ptr_d, idx_d, val_d_half, vec_d_half, res_d_half, num_rows);
					half_to_float<<<64, 1024, 0, stream>>>(res_d_half, res_d, num_rows);
				} else {
					light_spmv<<<num_blocks, block_size_1d, block_size_1d * sizeof(int_type), stream>>>(row_counter, ptr_d, idx_d, val_d, vec_d, res_d, num_rows);
				}
				break;
			default:
				if (debug) std::cout << "invalid spmv gpu implementation selected:" << gpu_impl << std::endl;
				exit(-1);
		}
		cudaDeviceSynchronize();
		// Read back the results from the device to verify the output;
		cudaMemcpy(res, res_d, sizeof(real_type) * num_rows, cudaMemcpyDefault);	

		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Kernel terminated" << std::endl;
			std::cout << "Computation took " << elapsed / 1e6 << " ms" << std::endl;
		}
	}

	void read_result(real_type *res_in) {
		// Read output;
		memcpy(res_in, res, sizeof(real_type) * num_rows);
	}

	long reset(real_type *vec_, int debug) {
		auto start = clock_type::now();
		vec = vec_;
		row_counter[0] = 0;

		// Reset result vector;
		memset(res, 0, sizeof(real_type) * num_rows);

		cudaMemcpy(vec_d, vec, sizeof(real_type) * num_cols, cudaMemcpyDefault);
		cudaMemcpy(res_d, res, sizeof(real_type) * num_rows, cudaMemcpyDefault);
		if (use_half_precision_gpu) float_to_half<<<64, 1024>>>(vec_d, vec_d_half, num_cols);
		cudaDeviceSynchronize();
		
		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Reset took " << elapsed / 1e6 << " ms" << std::endl;
		}
		return elapsed;
	}
};

/////////////////////////////
/////////////////////////////

template<typename V>
long sw_test(int_type *ptr, int_type *idx, V *val, int_type rows, V *vec, V *sw_res) {
	auto start = clock_type::now();
	spmv_gold(ptr, idx, val, rows, sw_res, vec);
	auto end = clock_type::now();
	auto sw_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	return sw_time;
}

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

	Options options = Options(argc, argv);
	int debug = (int) options.debug;	
	bool reset = options.reset;
	int block_size_1d = options.block_size_1d;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1); // Uniform distribution between 0 and 1;

	int_type nnz;
	int_type rows;
	int_type cols; // Size of the dense vector multiplied by the matrix;

	std::vector<int_type> x;
	std::vector<int_type> y;
	std::vector<real_type> val_coo;

	int read_values = !options.ignore_matrix_values; // If false, all values in the matrix are = 1; Set it true only for non-graphs;
	auto start_1 = clock_type::now();
	readMtx(options.use_sample_matrix ? DEFAULT_MTX_FILE : options.matrix_path.c_str(), &x, &y, &val_coo, &rows, &cols, &nnz, 0, read_values, debug, true, false);

	// Convert the COO matrix to CSR;
	int_type *ptr;
	int_type *idx;
	real_type *val;
	posix_memalign((void**) &ptr, 4096, (rows + 1) * sizeof(int_type));
	posix_memalign((void**) &idx, 4096, nnz * sizeof(int_type));
	posix_memalign((void**) &val, 4096, nnz * sizeof(real_type));
	coo2csr(ptr, idx, val, x, y, val_coo, rows, cols, false);

	// Vector multiplied by the sparse matrix;
	real_type *vec;
	posix_memalign((void**) &vec, 4096, cols * sizeof(real_type));
	create_sample_vector(vec, cols, true, true);

	// Temporary output of hardware SpMV;
	std::vector<real_type> hw_res(rows, 0);

	auto end_1 = clock_type::now();
	auto loading_time = chrono::duration_cast<chrono::milliseconds>(end_1 - start_1).count();

	if (debug) {
		std::cout << "loaded matrix with " << rows << " rows, " << cols << " columns and " << nnz << " non-zero elements" << std::endl;
		std::cout << "setup time=" << loading_time << " ms" << std::endl;
	}

   	//////////////////////////////
	// Generate software result //
	//////////////////////////////

	std::vector<real_type> sw_res(rows, 0);

	// Output of software SpMV, it contains all the similarities for all documents;
	auto sw_time = sw_test(ptr, idx, val, rows, vec, sw_res.data());

	if (debug) {
		std::cout << "\nsw results =" << std::endl;
		print_array_indexed(sw_res);
		std::cout << "sw time=" << sw_time << " ms" << std::endl;
	}

	/////////////////////////////
	// Setup hardware ///////////
	/////////////////////////////

	auto start_4 = clock_type::now();
	SpMV spmv(ptr, idx, val, rows, cols, nnz, vec, block_size_1d, debug, options.gpu_impl, options.use_half_precision_gpu);
	auto end_4 = clock_type::now();
	auto gpu_setup_time = chrono::duration_cast<chrono::milliseconds>(end_4 - start_4).count();
	if (debug) {
		std::cout << "gpu setup time=" << gpu_setup_time << " ms" << std::endl;
	}

	/////////////////////////////
	// Execute the kernel ///////
	/////////////////////////////

	uint num_tests = options.num_tests;
	std::vector<float> exec_times;

	for (uint i = 0; i < num_tests; i++) {

		if (debug) {
			std::cout << "\nIteration " << i << ")" << std::endl;
		}
		// Create a new input vector and compute SW results;
		if (reset) {
			create_sample_vector(vec, cols, true, false, true);
			sw_time = sw_test(ptr, idx, val, rows, vec, sw_res.data());		
		}
		// Reset the computation at each iteration;
		spmv.reset(vec, debug);

		auto start_5 = clock_type::now();
		// Main GPU computation;
		spmv(debug);
		auto end_5 = clock_type::now();
		// Retrieve results;
		spmv.read_result(hw_res.data());
		auto gpu_exec_time = chrono::duration_cast<chrono::milliseconds>(end_5 - start_5).count();
		exec_times.push_back(gpu_exec_time);

		int error = check_array_equality(hw_res.data(), sw_res.data(), rows, 10e-5, debug);

		//////////////////////////////
		// Check correctness /////////
		//////////////////////////////
		if (debug) {
			std::cout << "hw results=" << std::endl;
			print_array_indexed(hw_res);
			std::cout << "num errors = " << error << std::endl;
			std::cout << "gpu exec time=" << gpu_exec_time << " ms" << std::endl;
		} else {
			if(i == 0) {
				std::cout << "iteration,num_errors,sw_time_ms,hw_setup_time_ms,hw_exec_time_ms" << std::endl;
			}
			std::cout << i << "," << error << "," << sw_time << "," << gpu_setup_time << "," << gpu_exec_time << std::endl;
		}
	}
	// Print summary of results;
	if (debug) {
		int old_precision = std::cout.precision();
		std::cout.precision(2);
		std::cout << "----------------" << std::endl;
		std::cout << "Mean GPU execution time=" << mean(exec_times) << "±"
				<< st_dev(exec_times) << " ms" << std::endl;
		std::cout << "----------------" << std::endl;
		std::cout.precision(old_precision);
	}
}
