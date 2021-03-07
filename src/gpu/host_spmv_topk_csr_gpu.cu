#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <chrono>
#include <random>
#include <cusparse.h> 

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

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

template <typename V>
__global__ void get_topk(int_type *indices, V *res, V *topk_res, int k) {
	int k_i = blockIdx.x * gridDim.x + threadIdx.x;
	if (k_i < k) {
		topk_res[k_i] = res[indices[k - 1 - k_i]];
	}
}

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

	int k;

	cudaStream_t stream;
	cusparseHandle_t handle;

	cusparseSpMatDescr_t matrix;
    cusparseDnVecDescr_t vec_cusparse, res_cusparse;
    void *cusparse_buffer;
    size_t buffer_size;

	// SpMV cuSPARSE coefficients, no need to change them;
    real_type alpha = 1.0f;
	real_type beta = 0.0f;
	
	// Device array with values from 0 to rows - 1, used for arg-sort to indentify rows;
	thrust::device_vector<int_type> index;
	int_type *res_topk_idx;
	real_type *res_topk_d;
	half *res_topk_d_half;

	GPU_IMPL gpu_impl;
	bool use_half_precision_gpu;

	SpMV(int_type *ptr_, int_type *idx_, real_type *val_, int_type num_rows_, int_type num_cols_, int_type num_nnz_, real_type *vec_, int k, int block_size_1d = DEFAULT_BLOCK_SIZE_1D, int debug = 0, GPU_IMPL gpu_impl=GPU_IMPL(0), bool use_half_precision_gpu=DEFAULT_USE_HALF_PRECISION_GPU) :
			ptr(ptr_), idx(idx_), val(val_), num_rows(num_rows_), num_cols(num_cols_), num_nnz(num_nnz_), vec(vec_), block_size_1d(block_size_1d), k(k), gpu_impl(GPU_IMPL(gpu_impl)), use_half_precision_gpu(use_half_precision_gpu) {
		// Compute number of blocks required in the computation;
		num_blocks = ceil(num_rows / (float) block_size_1d);
		
		// Transfer data;
		setup(debug);
	}

	void setup(int debug) {

		// Device array with values from 0 to rows - 1, used for arg-sort to indentify rows;
		index = thrust::device_vector<int_type>(num_rows);
		thrust::sequence(index.begin(), index.end());
		res_topk_idx = (int_type*) calloc(k, sizeof(int_type));

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
		cudaMallocManaged(&res_topk_d, sizeof(real_type) * k);
		cudaMalloc(&res_topk_d_half, sizeof(half) * k);
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

	float operator()(int debug) {
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
				} else {
					light_spmv<<<num_blocks, block_size_1d, block_size_1d * sizeof(int_type), stream>>>(row_counter, ptr_d, idx_d, val_d, vec_d, res_d, num_rows);
				}
				break;
			default:
				if (debug) std::cout << "invalid spmv gpu implementation selected:" << gpu_impl << std::endl;
				exit(-1);
		}
		cudaDeviceSynchronize();
		float duration_spmv_only =  chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();

		// Arg-sort over results;
		if (use_half_precision_gpu && gpu_impl == CSR_LIGHTSPMV) {
			thrust::device_ptr<half> res_ptr(res_d_half);
			thrust::device_vector<half> res_vec(res_ptr, res_ptr + num_rows);
			auto key = thrust::make_permutation_iterator(thrust::make_transform_iterator(res_vec.begin(), thrust::identity<thrust::tuple<half>>{}), index.begin());
			thrust::sort_by_key(key, thrust::next(key, index.size()), index.begin());
	
			// Read back the results from the device to verify the output;
			int_type *raw_index_ptr = thrust::raw_pointer_cast(index.data() + num_rows - k);
			cudaMemcpyAsync(res_topk_idx, raw_index_ptr, sizeof(int_type) * k, cudaMemcpyDeviceToHost, stream);
			get_topk<<<1, 1024, 0, stream>>>(raw_index_ptr, res_d_half, res_topk_d_half, k);
			half_to_float<<<64, 1024, 0, stream>>>(res_topk_d_half, res_topk_d, k);
		} else {
			thrust::device_ptr<real_type> res_ptr(res_d);
			thrust::device_vector<real_type> res_vec(res_ptr, res_ptr + num_rows);
			auto key = thrust::make_permutation_iterator(thrust::make_transform_iterator(res_vec.begin(), thrust::identity<thrust::tuple<real_type>>{}), index.begin());
			thrust::sort_by_key(key, thrust::next(key, index.size()), index.begin());
	
			// Read back the results from the device to verify the output;
			int_type *raw_index_ptr = thrust::raw_pointer_cast(index.data() + num_rows - k);
			cudaMemcpyAsync(res_topk_idx, raw_index_ptr, sizeof(int_type) * k, cudaMemcpyDeviceToHost, stream);
			get_topk<<<1, 1024, 0, stream>>>(raw_index_ptr, res_d, res_topk_d, k);
		}
		
		// cudaMemcpyAsync(res, res_d, sizeof(real_type) * num_rows, cudaMemcpyDeviceToHost, stream);
		// Wait for computation to end;
		cudaDeviceSynchronize();

		float elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Kernel terminated" << std::endl;
			std::cout << "Computation took " << elapsed / 1e6 << " ms, spmv=" << duration_spmv_only / 1e6 << " ms, sorting=" << (elapsed - duration_spmv_only) / 1e6 << std::endl;
		}
		return duration_spmv_only;
	}

	void read_result(std::vector<real_type> &res_, std::vector<int_type> &res_idx_, int debug=0) {
		// Read output;
		for (int i = 0; i < k; i++) {
			res_idx_[i] = res_topk_idx[k - 1 - i];
			res_[i] = res_topk_d[i];
		}
	}

	long reset(real_type *vec_, int debug) {
		auto start = clock_type::now();
		vec = vec_;
		row_counter[0] = 0;

		// Fill index array;
		thrust::sequence(index.begin(), index.end());

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

template<typename I, typename V>
std::tuple<float, float> sw_test(coo_t<I, V> &coo, std::vector<V> &sw_res, std::vector<int_type> &res_idx_sw, std::vector<V> &res_sim_sw, V *vec, int_type top_k_value) {
	auto start_2 = clock_type::now();
	// spmv_coo_gold4(coo, sw_res.data(), vec);
	// Sort results and keep the top-K;
	// std::vector<int_type> sw_res_idx = sort_pr(sw_res.size(), sw_res.data());
	auto end_2 = clock_type::now();
	float sw_time_1 = (float) chrono::duration_cast<chrono::microseconds>(end_2 - start_2).count() / 1000;

	auto start_3 = clock_type::now();
	spmv_coo_gold_top_k(coo, vec, top_k_value, res_idx_sw.data(), res_sim_sw.data());

	// Sort the K output values;
	sort_tuples(top_k_value, res_idx_sw.data(), res_sim_sw.data());
	auto end_3 = clock_type::now();
	float sw_time_2 = (float) chrono::duration_cast<chrono::microseconds>(end_3 - start_3).count() / 1000;

	return std::make_tuple(sw_time_1, sw_time_2);
}

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

	// thrust::host_vector<float> H(4);
	// thrust::host_vector<int> I(4);
    // H[0] = 0.2;
    // H[1] = 0.1;
    // H[2] = 0.4;
	// H[3] = 0.01;
	// thrust::device_vector<float> D = H;
	// thrust::device_vector<int> index{D.size()};
	// thrust::sequence(index.begin(), index.end());
	// auto key = thrust::make_permutation_iterator(thrust::make_transform_iterator(D.cbegin(), thrust::identity<thrust::tuple<float>>{}), index.cbegin());
    // thrust::sort_by_key(key, thrust::next(key, index.size()), index.begin());
	// thrust::copy(index.cbegin(), index.cend(), I.begin());

	Options options = Options(argc, argv);
	int debug = (int) options.debug;	
	bool reset = options.reset;
	int block_size_1d = options.block_size_1d;
	int top_k_value = options.top_k_value;

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
	// Wrap the COO matrix;
	coo_t<int_type, real_type> coo = coo_t<int_type, real_type>(x, y, val_coo);
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

	// Output of software SpMV, it contains all the similarities for all documents;
	std::vector<real_type> sw_res(coo.num_rows, 0);
	std::vector<real_type> res_sim_sw(top_k_value, 0);
	std::vector<int_type> res_idx_sw(top_k_value, 0);

	// Output of software SpMV, it contains all the similarities for all documents;
	std::tuple<float, float> sw_time = sw_test(coo, sw_res, res_idx_sw, res_sim_sw, vec, top_k_value);
	float sw_time_1 = std::get<0>(sw_time);
	float sw_time_2 = std::get<1>(sw_time);

	if (debug) {
		std::cout << "\nsw results =" << std::endl;
		for (int i = 0; i < top_k_value; i++) {
			std::cout << i << ") document " << res_idx_sw[i] << " = " << res_sim_sw[i] << std::endl;
		}
		std::cout << "sw errors = " << check_array_equality(sw_res.data(), res_sim_sw.data(), 10e-6, top_k_value, true) << std::endl;
		std::cout << "sw time, full matrix=" << sw_time_1 << " ms; sw time, top-k=" << sw_time_2 << " ms" << std::endl;
	}

	/////////////////////////////
	// Setup hardware ///////////
	/////////////////////////////

	auto start_4 = clock_type::now();
	SpMV spmv(ptr, idx, val, rows, cols, nnz, vec, top_k_value, block_size_1d, debug, options.gpu_impl, options.use_half_precision_gpu);
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
	std::vector<float> readback_times;

	for (uint i = 0; i < num_tests; i++) {

		if (debug) {
			std::cout << "\nIteration " << i << ")" << std::endl;
		}
		// Create a new input vector and compute SW results;
		if (reset) {
			create_sample_vector(vec, cols, true, false, true);
			std::tuple<float, float> sw_time = sw_test(coo, sw_res, res_idx_sw, res_sim_sw, vec, top_k_value);
			sw_time_1 = std::get<0>(sw_time);
			sw_time_2 = std::get<1>(sw_time);
		}
		// Reset the computation at each iteration;
		spmv.reset(vec, debug);

		// Final output of hardware SpMV, it contains only the Top-K similarities and the Top-K indices;
		std::vector<real_type> hw_res(top_k_value);
		std::vector<int_type> hw_res_idx(top_k_value);

		auto start_5 = clock_type::now();
		// Main GPU computation;
		float spmv_only_time = spmv(debug) / 1e6;
		auto end_5 = clock_type::now();
		float gpu_exec_time = (float) chrono::duration_cast<chrono::nanoseconds>(end_5 - start_5).count() / 1e6;
		exec_times.push_back(gpu_exec_time);

		// Retrieve results;
		auto start_6 = clock_type::now();
		spmv.read_result(hw_res, hw_res_idx, debug);
		auto end_6 = clock_type::now();
		float readback_time = (float) chrono::duration_cast<chrono::nanoseconds>(end_6 - start_6).count() / 1e6;
		readback_times.push_back(readback_time);

		//////////////////////////////
		// Check correctness /////////
		//////////////////////////////
		int res_size = (int) hw_res_idx.size();
		if (debug) std::cout << "errors on indices =" << std::endl;
		int error_idx = check_array_equality(hw_res_idx.data(), res_idx_sw.data(), std::min(top_k_value, res_size), 0, debug);
		if (debug) std::cout << "errors on values =" << std::endl;
		int error = check_array_equality(hw_res.data(), res_sim_sw.data(), std::min(top_k_value, res_size), 10e-6, debug);
		if (debug) {
			std::cout << "sw results =" << std::endl;
			for (int j = 0; j < top_k_value; j++) {
				std::cout << j << ") document " << res_idx_sw[j] << " = " << res_sim_sw[j] << std::endl;
			}
			std::cout << "hw results=" << std::endl;
			for (int j = 0; j < std::min(top_k_value, res_size); j++) {
				std::cout << j << ") document " << hw_res_idx[j] << " = " << hw_res[j] << std::endl;
			}
			std::cout << "num errors on indices=" << error_idx << std::endl;
			std::cout << "num errors on values=" << error << std::endl;
			std::cout << "gpu exec time=" << gpu_exec_time << " ms" << std::endl;
		} else {
			if(i == 0) {
				std::cout << "iteration,error_idx,error_val,sw_full_time_ms,sw_topk_time_ms,hw_setup_time_ms,hw_spmv_only_time_ms,hw_exec_time_ms,readback_time_ms,k,sw_res_idx,sw_res_val,hw_res_idx,hw_res_val" << std::endl;
			}
			std::string sw_res_idx_str = "";
			std::string sw_res_val_str = "";
			std::string hw_res_idx_str = "";
			std::string hw_res_val_str = "";
			for (int j = 0; j < res_idx_sw.size(); j++) {
				sw_res_idx_str += std::to_string(res_idx_sw[j]) + ((j < res_idx_sw.size() - 1) ? ";" : "");
				sw_res_val_str += std::to_string(res_sim_sw[j]) + ((j < res_sim_sw.size() - 1) ? ";" : "");
			}
			for (int j = 0; j < hw_res_idx.size(); j++) {
				hw_res_idx_str += std::to_string(hw_res_idx[j]) + ((j < hw_res_idx.size() - 1) ? ";" : "");
				hw_res_val_str += std::to_string(hw_res[j]) + ((j < hw_res.size() - 1) ? ";" : "");
			}
			std::cout << i << "," << error_idx << "," << error << "," << sw_time_1 << "," << sw_time_2 << "," << gpu_setup_time << "," << spmv_only_time << "," << gpu_exec_time << "," << readback_time << "," << top_k_value << "," <<
					sw_res_idx_str << "," << sw_res_val_str << "," << hw_res_idx_str << "," << hw_res_val_str << std::endl;
		}
	}
	// Print summary of results;
	if (debug) {
		int old_precision = std::cout.precision();
		std::cout.precision(4);
		std::cout << "----------------" << std::endl;
		std::cout << "Mean FPGA execution time=" << mean(exec_times, 2) << "±" << st_dev(exec_times, 2) << " ms" << std::endl;
		std::cout << "Mean read-back time=" << mean(readback_times, 2) << "±" << st_dev(exec_times, 2) << " ms" << std::endl;
		std::cout << "----------------" << std::endl;
		std::cout.precision(old_precision);
	}
}
