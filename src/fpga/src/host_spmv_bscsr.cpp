
#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <chrono>
#include <random>
#include <CL/cl_ext.h>
#include <unordered_set>

#include "opencl_utils.hpp"
#include "../../common/utils/utils.hpp"
#include "../../common/utils/options.hpp"
#include "../../common/utils/evaluation_utils.hpp"
#include "../../common/types.hpp"
#include "ip/fpga_types.hpp"
#include "ip/coo_matrix.hpp"
#include "gold_algorithms/gold_algorithms.hpp"
#include "ip/fpga_utils.hpp"
#include "aligned_allocator.h"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

struct SubSpMVPartition {
	int_type partition_id;
	std::vector<std::tuple<int_type, int_type, real_type_inout>> coo_partition;
	std::vector<input_block, aligned_allocator<input_block>> coo_in;
	std::vector<input_packet_int_bscsr, aligned_allocator<input_packet_int_bscsr>> res_idx_out;
	std::vector<input_packet_real_inout_bscsr, aligned_allocator<input_packet_real_inout_bscsr>> res_out;

	cl::Buffer res_idx_buf;
	cl::Buffer res_buf;

	// Each partition has a fixed number of rows and a number of nnz that depends on the rows.
	// We track the first and last rows associated with the partition to split the total COO nnz;
	int_type num_rows_partition;
	int_type num_nnz_partition;
	int_type first_row;
	int_type last_row;

	int_type num_blocks_rows;
	int_type num_blocks_nnz;

	SubSpMVPartition(int_type _num_rows_partition, int_type _partition_id): num_rows_partition(_num_rows_partition), partition_id(_partition_id) {
		num_blocks_rows = (num_rows_partition + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;
		for (int_type i = 0; i < K * TOPK_RES_COPIES; i++) {
			input_packet_int_bscsr packet_res_idx;
			input_packet_real_inout_bscsr packet_res;
			res_idx_out.push_back(packet_res_idx);
			res_out.push_back(packet_res);
		}
	}
};

struct SuperSpMVPartition {
	int_type partition_id;
	cl::Event write_event;
	cl::Event reset_event;
	cl::Event computation_event;
	cl::Event readback_event;
	cl::Kernel kernel;
	cl::Buffer vec_buf;
	std::vector<SubSpMVPartition> partitions;

	SuperSpMVPartition(int_type _partition_id, std::vector<int_type> &_num_rows_partition, cl::Kernel _kernel): partition_id(_partition_id), kernel(_kernel) {
		for (int i = 0; i < SUB_SPMV_PARTITIONS; i++) {
			partitions.push_back(SubSpMVPartition(_num_rows_partition[i], partition_id * SUB_SPMV_PARTITIONS + i));
		}
	}
};

struct SpMV {
	ConfigOpenCL config;

	int_type *x;
	int_type *y;
	real_type_inout *val;

	int_type num_rows;
	int_type num_cols;
	int_type num_nnz;

	real_type_inout *vec;
	vec_real_inout_bscsr *vec_in;

	int_type num_blocks_cols;
	int_type cols_padded;

	std::vector<SuperSpMVPartition> partitions;

	// Keep a copy of events;
	std::vector<cl::Event> write_events;
	std::vector<cl::Event> reset_events;
	std::vector<cl::Event> computation_events;
	std::vector<cl::Event> readback_events;

	SpMV(ConfigOpenCL &config_, int_type *x_, int_type *y_, real_type_inout *val_, int_type num_rows_, int_type num_cols_, int_type num_nnz_, real_type_inout *vec_, int debug=0) :
			config(config_), x(x_), y(y_), val(val_), num_rows(num_rows_), num_cols(num_cols_), num_nnz(num_nnz_), vec(vec_) {

		num_blocks_cols = (num_cols + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;
		cols_padded = num_blocks_cols * BSCSR_PACKET_SIZE;
		posix_memalign((void**) &vec_in, 4096, num_blocks_cols * sizeof(vec_real_inout_bscsr));

		// Initialize partitions;
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			std::vector<int_type> num_rows_partitions;
			for (int_type j = 0; j < SUB_SPMV_PARTITIONS; j++) {
				int idx = i * SUB_SPMV_PARTITIONS + j;
				// Each partition has the same amount of rows, but the last partition might have fewer rows;
				num_rows_partitions.push_back((idx == SPMV_PARTITIONS - 1) ? (num_rows / SPMV_PARTITIONS) : ((num_rows + SPMV_PARTITIONS - 1) / SPMV_PARTITIONS));
			}
			SuperSpMVPartition p(i, num_rows_partitions, config.kernel[i]);
			partitions.push_back(p);
		}
		// Create the COO data structure as a sequence of packets of (x, y, val);
		packet_coo(debug);
		setup(debug);
	}

	SubSpMVPartition& get_partition(int_type i) {
		int_type super = i / SUB_SPMV_PARTITIONS;
		int_type sub = i % SUB_SPMV_PARTITIONS;
		return partitions[super].partitions[sub];
	}

	void packet_coo(int debug=0) {

		// First, split the COO;
		int_type num_rows_per_partition = (num_rows + SPMV_PARTITIONS - 1) / SPMV_PARTITIONS;
		for (int_type i = 0; i < num_nnz; i++) {
			// Find the partition of this entry;
			int_type curr_p = x[i] / num_rows_per_partition;
			get_partition(curr_p).coo_partition.push_back(std::tuple<int_type, int_type, real_type_inout>(x[i], y[i], val[i]));
		}

		// Book-keeping;
		for (int_type i = 0; i < SPMV_PARTITIONS; i++) {
			get_partition(i).first_row = std::get<0>(get_partition(i).coo_partition[0]);
			get_partition(i).last_row = std::get<0>(get_partition(i).coo_partition[get_partition(i).coo_partition.size() - 1]);
			get_partition(i).num_nnz_partition = get_partition(i).coo_partition.size();
			get_partition(i).num_blocks_nnz = (get_partition(i).num_nnz_partition + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;
			if (debug) std::cout << "partition " << i << ") size=" << get_partition(i).num_nnz_partition << "; blocks=" << get_partition(i).num_blocks_nnz << "; start=" << get_partition(i).first_row << "; end=" << get_partition(i).last_row << std::endl;
		}

		// Packet each COO partition;
		int_type r_last = 0;
		for (int_type i = 0; i < SPMV_PARTITIONS; i++) {
			packet_coo_partition(get_partition(i), r_last);
			r_last = get_partition(i).last_row;
		}

		// Packet the input vector;
//		for (int_type i = 0; i < num_blocks_cols; i++) {
//			input_packet_real_inout_bscsr new_block;
//			for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
//				int_type index = j + BSCSR_PACKET_SIZE * i;
//				if (index < num_cols) {
//					new_block[j] = vec[index];
//				} else {
//					new_block[j] = 0;
//				}
//			}
//			vec_in[i] = new_block;
//		}

		real_type_inout vec_buffer[BSCSR_PACKET_SIZE];
		for (int_type i = 0; i < num_blocks_cols; i++) {
			vec_real_inout_bscsr new_block_vec;
			for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
				int_type index = j + BSCSR_PACKET_SIZE * i;
				if (index < num_cols) {
					vec_buffer[j] = vec[index];
				} else {
					vec_buffer[j] = 0;
				}
			}
			write_block_vec(&new_block_vec, vec_buffer);
			vec_in[i] = new_block_vec;
		}
	}

	void packet_coo_partition(SubSpMVPartition &p, int_type last_r) {
		int_type curr_row = 0;
		int_type x_local[BSCSR_PACKET_SIZE];
		int_type y_local[BSCSR_PACKET_SIZE];
		real_type_inout val_local[BSCSR_PACKET_SIZE];
		bool_type xf_local[1];
		curr_row = last_r;
		for (int_type i = 0; i < p.num_blocks_nnz; i++) {
			input_block tmp_block_512;
			if (std::get<0>(p.coo_partition[BSCSR_PACKET_SIZE * i]) != curr_row){
				xf_local[0] = (bool_type) true;
				write_block_xf(&tmp_block_512, xf_local);
				curr_row = std::get<0>(p.coo_partition[BSCSR_PACKET_SIZE * i]);
			} else {
				xf_local[0] = (bool_type) false;
				write_block_xf(&tmp_block_512, xf_local);
			}
			for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
				int_type index = j + BSCSR_PACKET_SIZE * i;
				auto curr_tuple = p.coo_partition[index];
				curr_row = std::get<0>(curr_tuple);
				if (index < p.num_nnz_partition) {
					x_local[j] = 0;
					y_local[j]= std::get<1>(curr_tuple);
					val_local[j] = std::get<2>(curr_tuple);
				} else {
					x_local[j] = 0;
					y_local[j] = 0;
					val_local[j] = 0;
				}
			}
			int_type pos = 0;
			int_type same_row_values = 1;
			for (int_type j = 1; j < BSCSR_PACKET_SIZE; j++) {
				if (j - 1 + (BSCSR_PACKET_SIZE * i) < p.num_nnz_partition){
					if (std::get<0>(p.coo_partition[j + (BSCSR_PACKET_SIZE * i)]) == std::get<0>(p.coo_partition[j - 1 + (BSCSR_PACKET_SIZE * i)])) {
						same_row_values++;
					} else {
						x_local[pos] = same_row_values;
						same_row_values = 1;
						pos++;
					}
				} else {
					x_local[pos] = 0;
					pos++;
				}
			}
			if (BSCSR_PACKET_SIZE - 1 + (BSCSR_PACKET_SIZE * i) < p.num_nnz_partition){
				x_local[pos] = same_row_values;
			}
			// Accumulate values in x;
			for (int_type j = 1; j < BSCSR_PACKET_SIZE; j++) {
				x_local[j] += x_local[j - 1];
			}
			write_block_x(&tmp_block_512, x_local);
			write_block_y(&tmp_block_512, y_local);
			write_block_val(&tmp_block_512, val_local);
			p.coo_in.push_back(tmp_block_512);
		}
	}

	void setup(int debug) {

		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug) std::cout << "Create Kernel Arguments (partition " << i << ")" << std::endl;

			int a = i * SUB_SPMV_PARTITIONS + 0;
			int b = i * SUB_SPMV_PARTITIONS + 1;
			int c = i * SUB_SPMV_PARTITIONS + 2;
			int d = i * SUB_SPMV_PARTITIONS + 3;

			// Create the input and output arrays in device memory
			cl::Buffer coo_buf0 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(a).num_blocks_nnz, get_partition(a).coo_in.data());
			cl::Buffer coo_buf1 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(b).num_blocks_nnz, get_partition(b).coo_in.data());
			cl::Buffer coo_buf2 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(c).num_blocks_nnz, get_partition(c).coo_in.data());
			cl::Buffer coo_buf3 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(d).num_blocks_nnz, get_partition(d).coo_in.data());

			partitions[i].vec_buf = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(vec_real_inout_bscsr) * num_blocks_cols, vec_in);
			get_partition(a).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(a).res_idx_out.data());
			get_partition(b).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(b).res_idx_out.data());
			get_partition(c).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(c).res_idx_out.data());
			get_partition(d).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(d).res_idx_out.data());
			get_partition(a).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(a).res_out.data());
			get_partition(b).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(b).res_out.data());
			get_partition(c).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(c).res_out.data());
			get_partition(d).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(d).res_out.data());

			// Set kernel arguments
			int narg = 0;
			partitions[i].kernel.setArg(narg++, coo_buf0);
			partitions[i].kernel.setArg(narg++, coo_buf1);
			partitions[i].kernel.setArg(narg++, coo_buf2);
			partitions[i].kernel.setArg(narg++, coo_buf3);

			partitions[i].kernel.setArg(narg++, get_partition(a).num_rows_partition);
			partitions[i].kernel.setArg(narg++, get_partition(b).num_rows_partition);
			partitions[i].kernel.setArg(narg++, get_partition(c).num_rows_partition);
			partitions[i].kernel.setArg(narg++, get_partition(d).num_rows_partition);

			partitions[i].kernel.setArg(narg++, num_cols);
			partitions[i].kernel.setArg(narg++, num_cols);
			partitions[i].kernel.setArg(narg++, num_cols);
			partitions[i].kernel.setArg(narg++, num_cols);

			partitions[i].kernel.setArg(narg++, get_partition(a).num_nnz_partition);
			partitions[i].kernel.setArg(narg++, get_partition(b).num_nnz_partition);
			partitions[i].kernel.setArg(narg++, get_partition(c).num_nnz_partition);
			partitions[i].kernel.setArg(narg++, get_partition(d).num_nnz_partition);

			partitions[i].kernel.setArg(narg++, partitions[i].vec_buf);

			partitions[i].kernel.setArg(narg++, get_partition(a).res_idx_buf);
			partitions[i].kernel.setArg(narg++, get_partition(b).res_idx_buf);
			partitions[i].kernel.setArg(narg++, get_partition(c).res_idx_buf);
			partitions[i].kernel.setArg(narg++, get_partition(d).res_idx_buf);

			partitions[i].kernel.setArg(narg++, get_partition(a).res_buf);
			partitions[i].kernel.setArg(narg++, get_partition(b).res_buf);
			partitions[i].kernel.setArg(narg++, get_partition(c).res_buf);
			partitions[i].kernel.setArg(narg++, get_partition(d).res_buf);

			// Transfer data from host to device (0 means host-to-device transfer);
			if (debug) std::cout << "Write inputs into device memory (partition " << i << ")" << std::endl;

			// Transfer data from host to device (0 means host-to-device transfer);
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects( { coo_buf0, coo_buf1, coo_buf2, coo_buf3, partitions[i].vec_buf }, 0, NULL, &partitions[i].write_event);
			write_events.push_back(partitions[i].write_event);

		}
		// Wait for completion of transfer;
		cl::Event::waitForEvents(write_events);
		write_events.clear();
	}

	long operator()(int debug) {
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug)  std::cout << "Execute the kernel (partition " << i << ")" << std::endl;

//			cl::Event e;
			config.queue[i % OPENCL_QUEUES].enqueueTask(partitions[i].kernel, NULL, &partitions[i].computation_event);
			computation_events.push_back(partitions[i].computation_event);
		}
//			int a = i * SUB_SPMV_PARTITIONS + 0;
//			int b = i * SUB_SPMV_PARTITIONS + 1;
//			int c = i * SUB_SPMV_PARTITIONS + 2;
//			int d = i * SUB_SPMV_PARTITIONS + 3;
//
//			// Read back the results from the device to verify the output
//			std::vector<cl::Event> events2({ e });
//			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects({
//				get_partition(a).res_idx_buf,
//				get_partition(b).res_idx_buf,
//				get_partition(c).res_idx_buf,
//				get_partition(d).res_idx_buf,
//				get_partition(a).res_buf,
//				get_partition(b).res_buf,
//				get_partition(c).res_buf,
//				get_partition(d).res_buf
//			}, CL_MIGRATE_MEM_OBJECT_HOST, &events2, &partitions[i].readback_event);
//			readback_events.push_back(partitions[i].readback_event);
//		}
		// Wait for computation to end;
		return wait(debug);
	}

	long wait(int debug) {
		auto start = clock_type::now();
		// Wait for completion of computation and read-back;
		cl::Event::waitForEvents(computation_events);
		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Kernel terminated" << std::endl;
			std::cout << "Computation took " << elapsed / 1e6 << " ms" << std::endl;
		}

		// Read-back;
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug)  std::cout << "Read result from kernel (partition " << i << ")" << std::endl;

			int a = i * SUB_SPMV_PARTITIONS + 0;
			int b = i * SUB_SPMV_PARTITIONS + 1;
			int c = i * SUB_SPMV_PARTITIONS + 2;
			int d = i * SUB_SPMV_PARTITIONS + 3;

			// Read back the results from the device to verify the output
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects({
				get_partition(a).res_idx_buf,
				get_partition(b).res_idx_buf,
				get_partition(c).res_idx_buf,
				get_partition(d).res_idx_buf,
				get_partition(a).res_buf,
				get_partition(b).res_buf,
				get_partition(c).res_buf,
				get_partition(d).res_buf
			}, CL_MIGRATE_MEM_OBJECT_HOST, 0, &partitions[i].readback_event);
			readback_events.push_back(partitions[i].readback_event);
		}
		start = clock_type::now();
		cl::Event::waitForEvents(readback_events);
		auto read_elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();

		if (debug) {
			std::cout << "Read-back took " << read_elapsed / 1e6 << " ms" << std::endl;
		}

		readback_events.clear();
		computation_events.clear();
		return elapsed;
	}

	void read_result(std::vector<real_type_inout> &res, std::vector<int_type> &res_idx, int debug=0) {

		// Store results in a map;
		std::unordered_map<int_type, real_type_inout> result_map;

		// Read output of each partition;
		for (int i = 0; i < SPMV_PARTITIONS; i++) {
			// Temporary vectors to store results for the current partition, used only for debug;
			std::vector<int_type> res_idx_tmp;
			std::vector<real_type_inout> res_tmp;

			for (int_type c = 0; c < TOPK_RES_COPIES; c++) {
				for (int_type j = 0; j < K; j++) {
					for (int_type q = 0; q < BSCSR_PACKET_SIZE; q++) {
						int_type index = q + BSCSR_PACKET_SIZE * (j + K * c);
						int_type index_2 = j + c * K;
						int_type tmp_idx = get_partition(i).res_idx_out[index_2][q] + get_partition(i).first_row; //We need to to add the starting row of the partition
						real_type_inout tmp_val = get_partition(i).res_out[index_2][q];
						if (tmp_val > 0) {  // Skip empty results;
							result_map.insert(std::pair<int_type, real_type_inout>(tmp_idx, tmp_val));
						}

						// Store results for each partition;
						if (debug) {
							res_idx_tmp.push_back(tmp_idx);
							res_tmp.push_back(tmp_val);
						}
					}
				}
			}

			if (debug) {
				sort_tuples(res_idx_tmp.size(), res_idx_tmp.data(), res_tmp.data());
				std::cout << "\nPartition " << i << " results:" << std::endl;
				for (int_type j = 0; j < res_idx_tmp.size(); j++) {
					if (res_tmp[j] == 0) {
						break;
					}
					std::cout << j << ") " << res_idx_tmp[j] << "=" << res_tmp[j] << std::endl;
				}
			}
		}

		int_type i = 0;
		for(auto it = result_map.begin(); it != result_map.end(); it++, i++) {
			res_idx.push_back(it->first);
			res.push_back(it->second);
		}
		sort_tuples(res.size(), res_idx.data(), res.data());
	}

	long reset(real_type_inout *vec_, int debug) {
		auto start = clock_type::now();
		vec = vec_;
		// Packet the input vector;
		real_type_inout vec_buffer[BSCSR_PACKET_SIZE];
		for (int_type i = 0; i < num_blocks_cols; i++) {
			vec_real_inout_bscsr new_block_vec;
			for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
				int_type index = j + BSCSR_PACKET_SIZE * i;
				if (index < num_cols) {
					vec_buffer[j] = vec[index];
				} else {
					vec_buffer[j] = 0;
				}
			}
			write_block_vec(&new_block_vec, vec_buffer);
			vec_in[i] = new_block_vec;
		}

		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			// Transfer data from host to device (0 means host-to-device transfer);
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects( { partitions[i].vec_buf }, 0, NULL, &partitions[i].reset_event);
			reset_events.push_back(partitions[i].reset_event);
		}

		// Wait for completion of transfer;
		cl::Event::waitForEvents(reset_events);
		reset_events.clear();

		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Reset took " << elapsed / 1e6 << " ms" << std::endl;
		}
		return elapsed;
	}
};

template<typename I, typename V>
std::tuple<float, float> sw_test(coo_t<I, V> &coo, std::vector<V> &sw_res, std::vector<int_type> &res_idx_sw, std::vector<V> &res_sim_sw, V *vec, int_type top_k_value) {
	auto start_2 = clock_type::now();
	spmv_coo_gold4(coo, sw_res.data(), vec);
	// Sort results and keep the top-K;
	std::vector<int_type> sw_res_idx = sort_pr(sw_res.size(), sw_res.data());
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

	Options options = Options(argc, argv);
	int debug = (int) options.debug;
	bool reset = options.reset;
	int seed = 0;  // If 0, don't use seed;
    cl::Context context;

    std::string xclbin_path = options.xclbin_path;
	std::vector<std::string> target_devices = { "xilinx_u280_xdma_201920_3" };
	std::vector<std::string> kernels = { xclbin_path };
	std::string kernel_name = "spmv_bscsr_top_k_main";

	//setup kernel
	ConfigOpenCL config(kernel_name, SUPER_SPMV_PARTITIONS, OPENCL_QUEUES);
	setup_opencl(config, target_devices, kernels, debug);

	int top_k_value = options.top_k_value;

	int_type nnz;
	int_type rows;
	int_type cols; // Size of the dense vector multiplied by the matrix;

	std::vector<int_type> x;
	std::vector<int_type> y;
	std::vector<real_type_inout> val;

	int read_values = !options.ignore_matrix_values; // If false, all values in the matrix are = 1; Set it true only for non-graphs;
	auto start_1 = clock_type::now();
	readMtx(options.use_sample_matrix ? DEFAULT_MTX_FILE : options.matrix_path.c_str(), &x, &y, &val, &rows, &cols, &nnz, 0, read_values, debug, true, false);
	// Wrap the COO matrix;
	coo_t<int_type, real_type_inout> coo = coo_t<int_type, real_type_inout>(x, y, val);
//	coo.print_coo(true);

	// Vector multiplied by the sparse matrix;
	real_type_inout *vec;
	posix_memalign((void**) &vec, 4096, cols * sizeof(real_type_inout));
	create_sample_vector(vec, cols, true, false, true, seed);
//	print_array_indexed(vec, cols);
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
	std::vector<real_type_inout> sw_res(coo.num_rows, 0);
	std::vector<real_type_inout> res_sim_sw(top_k_value, 0);
	std::vector<int_type> res_idx_sw(top_k_value, 0);

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
	SpMV spmv(config, coo.start.data(), coo.end.data(), coo.val.data(), rows, cols, nnz, vec, debug);
	auto end_4 = clock_type::now();
	float fpga_setup_time = (float) chrono::duration_cast<chrono::microseconds>(end_4 - start_4).count() / 1000;
	if (debug) {
		std::cout << "fpga setup time=" << fpga_setup_time << " ms" << std::endl;
	}

	/////////////////////////////
	// Execute the kernel ///////
	/////////////////////////////

	uint num_tests = options.num_tests;
	std::vector<float> exec_times_full;
	std::vector<float> exec_times;
	std::vector<float> readback_times;
	std::vector<float> error_count;
	std::vector<float> precision_vec;

	for (uint i = 0; i < num_tests; i++) {

		if (debug) {
			std::cout << "\nIteration " << i << ")" << std::endl;
		}
		// Create a new input vector;
		if (reset) {
			create_sample_vector(vec, cols, true, false, true);
			std::tuple<float, float> sw_time = sw_test(coo, sw_res, res_idx_sw, res_sim_sw, vec, top_k_value);
			sw_time_1 = std::get<0>(sw_time);
			sw_time_2 = std::get<1>(sw_time);
			// Load the new vec on FPGA;
			spmv.reset(vec, debug);
		}
		// Final output of hardware SpMV, it contains only the Top-K similarities and the Top-K indices;
		std::vector<real_type_inout> hw_res;
		std::vector<int_type> hw_res_idx;

		auto start_5 = clock_type::now();
		// Main FPGA computation;
		float fpga_exec_time = (float) spmv(debug) / 1e6;
		auto end_5 = clock_type::now();
		float fpga_full_exec_time = (float) chrono::duration_cast<chrono::nanoseconds>(end_5 - start_5).count() / 1e6;
		exec_times.push_back(fpga_exec_time);
		exec_times_full.push_back(fpga_full_exec_time);

		// Retrieve results;
		auto start_6 = clock_type::now();
		spmv.read_result(hw_res, hw_res_idx, debug);
		auto end_6 = clock_type::now();
		float readback_time = (float) chrono::duration_cast<chrono::microseconds>(end_6 - start_6).count() / 1000;
		readback_times.push_back(readback_time);

		//////////////////////////////
		// Check correctness /////////
		//////////////////////////////
		int res_size = (int) hw_res_idx.size();
		if (debug) std::cout << "errors on indices =" << std::endl;
		int error_idx = check_array_equality(hw_res_idx.data(), res_idx_sw.data(), std::min(top_k_value, res_size), 0, debug);
		error_count.push_back(error_idx);
		error_idx += std::max(0, top_k_value - res_size);
		if (debug) std::cout << "errors on values =" << std::endl;
		int error = check_array_equality(hw_res.data(), res_sim_sw.data(), std::min(top_k_value, res_size), 10e-6, debug);
		error += std::max(0, top_k_value - res_size);
		std::unordered_set<int> s(res_idx_sw.begin(), res_idx_sw.end());
		int_type size_intersection = count_if(hw_res_idx.begin(), hw_res_idx.end(), [&](int k) {return s.find(k) != s.end();});
		float precision = (((float) size_intersection)/((float) top_k_value));
		
		precision_vec.push_back(precision);

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
			std::cout << "precision=" << precision << std::endl;
			std::cout << "fpga exec time=" << fpga_exec_time << " ms, full exec time=" << fpga_full_exec_time << " ms" << std::endl;
		} else {
			if(i == 0) {
				std::cout << "iteration,error_idx,error_val,sw_full_time_ms,sw_topk_time_ms,hw_setup_time_ms,hw_exec_time_ms,hw_full_exec_time_ms,readback_time_ms,k,sw_res_idx,sw_res_val,hw_res_idx,hw_res_val" << std::endl;
			}
			std::string sw_res_idx_str = "";
			std::string sw_res_val_str = "";
			std::string hw_res_idx_str = "";
			std::string hw_res_val_str = "";
			for (int j = 0; j < res_idx_sw.size(); j++) {
				sw_res_idx_str += std::to_string(res_idx_sw[j]) + ((j < res_idx_sw.size() - 1) ? ";" : "");
#if USE_FLOAT
				sw_res_val_str += std::to_string(res_sim_sw[j]) + ((j < res_sim_sw.size() - 1) ? ";" : "");
#else
				sw_res_val_str += std::to_string(res_sim_sw[j].to_float()) + ((j < res_sim_sw.size() - 1) ? ";" : "");

#endif
			}
			for (int j = 0; j < hw_res_idx.size(); j++) {
				hw_res_idx_str += std::to_string(hw_res_idx[j]) + ((j < hw_res_idx.size() - 1) ? ";" : "");
#if USE_FLOAT
				hw_res_val_str += std::to_string(hw_res[j]) + ((j < hw_res.size() - 1) ? ";" : "");
#else
				hw_res_val_str += std::to_string(hw_res[j].to_float()) + ((j < hw_res.size() - 1) ? ";" : "");
#endif
			}
			std::cout << i << "," << error_idx << "," << error << "," << sw_time_1 << "," << sw_time_2 << "," << fpga_setup_time << "," << fpga_exec_time << "," << fpga_full_exec_time << ","  << readback_time << "," << top_k_value << "," <<
					sw_res_idx_str << "," << sw_res_val_str << "," << hw_res_idx_str << "," << hw_res_val_str << std::endl;
		}
	}
	// Print summary of results;
	if (debug) {
		int old_precision = cout.precision();
		cout.precision(4);
		std::cout << "----------------" << std::endl;
		std::cout << "Mean FPGA execution time=" << mean(exec_times, 2) << "±" << st_dev(exec_times, 2) << " ms" << std::endl;
		std::cout << "Mean full FPGA execution time=" << mean(exec_times_full, 2) << "±" << st_dev(exec_times_full, 2) << " ms" << std::endl;
		std::cout << "Mean read-back time=" << mean(readback_times, 2) << "±" << st_dev(exec_times, 2) << " ms" << std::endl;
		std::cout << "Mean error=" << mean(error_count, 2) << "±" << st_dev(error_count, 2) << std::endl;
		std::cout << "Mean precision=" << mean(precision_vec, 2) << "±" << st_dev(precision_vec, 2) << std::endl;
		std::cout << "----------------" << std::endl;
		cout.precision(old_precision);
	}
}


